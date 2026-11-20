"""Customer-aware constrained K-fold splitting for out-of-fold encodings.

Rows of the same customer (group) are always assigned to the same validation
fold, so no customer is ever split across folds. On top of that, every fold's
training side must contain at least ``min_records`` rows for every category of
the given ``cols``. Among all feasible partitions, the partition minimizing the
total, across validation folds and target labels, of

    |count(label in fold) - total(label) / n_splits|

is returned. Ties are broken deterministically so repeated calls with the same
data return the same partition.

The optimization is solved to global optimum as a mixed-integer linear program
via HiGHS (``scipy.optimize.milp``). When no feasible partition exists, an
:class:`InfeasibleSplitError` is raised explaining why; thresholds are never
silently lowered and groups are never split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from category_encoders import utils

__all__ = [
    'ConstrainedGroupKFold',
    'InfeasibleSplitError',
    'FoldCoverage',
    'ConstrainedSplitResult',
    'split_constrained_group_kfold',
]


class InfeasibleSplitError(ValueError):
    """Raised when no partition satisfies all fold constraints.

    The message always states which requirement cannot be met. The splitter
    never relaxes ``min_records`` and never splits a group to work around
    infeasibility.
    """


@dataclass
class FoldCoverage:
    """Per-fold coverage statistics for one constrained partition.

    Attributes
    ----------
    fold_id: int
        Zero-based fold index.
    n_val: int
        Number of rows on the validation side of the fold.
    n_train: int
        Number of rows on the training side of the fold.
    val_label_counts: dict
        Number of validation-side rows per target label.
    train_category_counts: dict of dict
        ``train_category_counts[col][category]`` is the number of training-side
        rows carrying that category of ``col``. NaN categories are keyed by
        ``numpy.nan``.
    min_train_category_counts: dict
        ``min_train_category_counts[col]`` is the smallest, over the column's
        categories, training-side count observed in this fold. Every value is
        at least the requested ``min_records`` when the partition is feasible.
    """

    fold_id: int
    n_val: int
    n_train: int
    val_label_counts: dict
    train_category_counts: dict
    min_train_category_counts: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Flatten the category coverage into a tidy DataFrame."""
        rows = []
        for col, per_category in self.train_category_counts.items():
            for category, count in per_category.items():
                rows.append(
                    {
                        'fold': self.fold_id,
                        'col': col,
                        'category': category,
                        'train_count': count,
                        'val_count': None,
                    }
                )
        for label, count in self.val_label_counts.items():
            rows.append(
                {
                    'fold': self.fold_id,
                    'col': '__label__',
                    'category': label,
                    'train_count': None,
                    'val_count': count,
                }
            )
        return pd.DataFrame(rows)


@dataclass
class ConstrainedSplitResult:
    """A feasible constrained partition together with its coverage statistics.

    Attributes
    ----------
    folds: list of numpy.ndarray
        ``folds[k]`` holds the positional row indices assigned to validation
        fold ``k``. Every fold is non-empty and the arrays partition
        ``range(n_rows)``.
    fold_assignment: numpy.ndarray
        ``fold_assignment[i]`` is the fold assigned to row ``i``. Rows of the
        same group share one value.
    groups: list
        Group labels in the canonical order used internally.
    group_folds: dict
        Fold assigned to every group label.
    coverage: list of FoldCoverage
        Coverage statistics, one entry per fold.
    objective_value: float
        The minimized total absolute deviation of per-fold, per-label
        validation counts from ``total(label) / n_splits``.
    cols: list of str
        Columns whose categories were protected by ``min_records``.
    min_records: int
        The enforced minimum training-side count.
    """

    folds: list
    fold_assignment: np.ndarray
    groups: list
    group_folds: dict
    coverage: list
    objective_value: float
    cols: list
    min_records: int

    @property
    def n_splits(self) -> int:
        """Number of validation folds."""
        return len(self.folds)

    def summary_frame(self) -> pd.DataFrame:
        """One-row-per-fold summary (sizes and per-column minimum counts)."""
        return pd.DataFrame(
            [
                {
                    'fold': cov.fold_id,
                    'n_val': cov.n_val,
                    'n_train': cov.n_train,
                    **{
                        f'min_train_count:{col}': minimum
                        for col, minimum in cov.min_train_category_counts.items()
                    },
                }
                for cov in self.coverage
            ]
        )

    def coverage_frame(self) -> pd.DataFrame:
        """Long-form coverage table concatenating every fold's statistics."""
        return pd.concat([cov.to_frame() for cov in self.coverage], ignore_index=True)


def _coerce_groups(groups, X, n_rows):
    """Return groups as a pandas Series aligned to X's rows."""
    if isinstance(groups, str):
        if not isinstance(X, pd.DataFrame) or groups not in X.columns:
            raise ValueError(f'groups={groups!r} names a column, but it was not found in X.')
        group_series = X[groups]
    else:
        group_series = pd.Series(np.asarray(groups))
    if len(group_series) != n_rows:
        raise ValueError(f'groups has length {len(group_series)} but X has {n_rows} rows.')
    if group_series.isna().any():
        raise ValueError('groups contains missing values; every row must belong to a named group.')
    return group_series


def _check_min_records(min_records):
    """Validate the min_records argument."""
    if isinstance(min_records, bool) or not isinstance(min_records, (int, np.integer)):
        raise TypeError(f'min_records must be a positive int, got {type(min_records).__name__}.')
    if min_records < 1:
        raise ValueError(f'min_records must be >= 1, got {min_records}.')
    return int(min_records)


class ConstrainedGroupKFold(BaseEstimator, BaseCrossValidator):
    """K-fold splitter that keeps groups intact and protects rare categories.

    Every group (e.g. a customer identifier) is assigned to exactly one
    validation fold, so a customer's rows are never separated. Each fold's
    *training* side is required to contain at least ``min_records`` rows for
    every category of every column in ``cols``. All ``n_splits`` validation
    folds are guaranteed non-empty.

    Among the feasible partitions, the global optimum of the label-balance
    objective is found with mixed-integer linear programming:

        minimize  sum over folds k and labels l of
                  |n_val(k, l) - n_total(l) / n_splits|

    Partitions tied on that objective are broken by a deterministic secondary
    criterion that assigns earlier groups, in the canonical group order, to
    earlier folds. The lexicographic objective and fixed solver seed make
    repeated calls on identical data return identical partitions.

    Parameters
    ----------
    n_splits: int
        Number of non-empty validation folds K.
    cols: str or list of str
        Column(s) whose categories must each appear at least ``min_records``
        times on every fold's training side.
    min_records: int, default=1
        Minimum number of training-side rows required for every protected
        category in every fold. Must be a positive integer; it is never
        relaxed automatically.
    groups_col: str, optional
        Name of the X column holding group identifiers. When set, ``split``
        can be called without an explicit ``groups`` argument.
    random_state: int, optional
        Seed forwarded to the HiGHS MILP solver. The solver is deterministic
        for a fixed seed; the default is already deterministic.

    Raises
    ------
    InfeasibleSplitError
        If the constraints cannot be satisfied. The error message reports the
        concrete reason.

    Examples
    --------
    >>> import pandas as pd
    >>> from category_encoders.cv_split import ConstrainedGroupKFold
    >>> X = pd.DataFrame({'cat': ['a', 'a', 'b', 'b', 'a', 'b']})
    >>> y = pd.Series([0, 1, 0, 1, 0, 1])
    >>> groups = [1, 1, 2, 2, 3, 3]
    >>> splitter = ConstrainedGroupKFold(n_splits=2, cols=['cat'], min_records=1)
    >>> folds = list(splitter.split(X, y, groups))
    >>> len(folds)
    2
    """

    def __init__(
        self,
        n_splits: int,
        cols,
        min_records: int = 1,
        groups_col: str | None = None,
        random_state: int | None = None,
    ):
        """Init the constrained group K-fold splitter."""
        if not isinstance(n_splits, (int, np.integer)) or isinstance(n_splits, bool):
            raise TypeError(f'n_splits must be an int, got {type(n_splits).__name__}.')
        if n_splits < 2:
            raise ValueError(f'n_splits must be >= 2, got {n_splits}.')
        self.n_splits = int(n_splits)
        # Keep the constructor argument untouched for sklearn's clone contract;
        # normalize to a list lazily where the splitter actually consumes it.
        self.cols = cols
        normalized = [cols] if isinstance(cols, str) else list(cols)
        if not normalized:
            raise ValueError('At least one column must be given in cols.')
        self._cols_list = normalized
        self.min_records = _check_min_records(min_records)
        self.groups_col = groups_col
        self.random_state = random_state
        # Filled by split(); the trailing underscore follows sklearn's
        # convention for attributes learned from data. Infeasibility is always
        # surfaced via InfeasibleSplitError, never stored silently.
        self.result_: ConstrainedSplitResult | None = None

    @property
    def cols_list(self) -> list:
        """Normalized list form of the protected ``cols`` argument."""
        return self._cols_list

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits

    def _prepare(self, X, y, groups):
        """Validate inputs and build the group/category/label count tables."""
        X, y = utils.convert_inputs(X, y)
        n_rows = len(X)
        missing = [col for col in self._cols_list if col not in X.columns]
        if missing:
            raise ValueError(f'Columns not found in X: {missing}.')
        if y is None:
            raise ValueError('y is required: the balance objective uses target labels.')
        if len(y) != n_rows:
            raise ValueError(f'y has length {len(y)} but X has {n_rows} rows.')
        # Crosstab aligns on pandas indexes; normalize everything to a shared
        # positional range so non-default or mismatched labels cannot misalign.
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        if groups is None and self.groups_col is not None:
            groups = self.groups_col
        if groups is None:
            raise ValueError('groups must be provided (or groups_col set at construction).')
        group_series = _coerce_groups(groups, X, n_rows).reset_index(drop=True)

        if n_rows < self.n_splits:
            raise InfeasibleSplitError(
                f'Only {n_rows} rows for {self.n_splits} non-empty folds; '
                'need at least one row per fold.'
            )

        # Canonical group order: factorize with sort=True gives a stable,
        # value-sorted code assignment independent of first appearance.
        group_codes, group_values = pd.factorize(group_series, sort=True)
        n_groups = len(group_values)
        if n_groups < self.n_splits:
            raise InfeasibleSplitError(
                f'Only {n_groups} group(s) for {self.n_splits} non-empty folds. '
                'Groups cannot be split, so each fold needs at least one group.'
            )

        # Rows per group, in canonical group order.
        group_sizes = np.bincount(group_codes, minlength=n_groups).astype(float)

        # Category counts per group for every protected column.
        # crosstab with dropna=False keeps NaN as an explicit category.
        category_tables = {}
        for col in self._cols_list:
            table = pd.crosstab(group_series, X[col], dropna=False)
            table = table.reindex(group_values, fill_value=0)
            category_tables[col] = table

        m = self.min_records
        # Structural feasibility pre-check with a concrete reason. The only
        # provably impossible pattern detectable without solving the partition
        # is a single group monopolizing a category: the fold hosting that
        # group excludes all of its rows at once. That fold's training side
        # can draw on at most (total - worst_group) rows of the category, so
        # it still needs at least min_records there. All other cases are
        # decided by the MILP, which proves infeasibility globally.
        for col, table in category_tables.items():
            totals = table.sum(axis=0)
            max_in_group = table.max(axis=0)
            for category, worst in max_in_group.items():
                worst = float(worst)
                total = float(totals[category])
                if total - worst < m:
                    offender = table[category].idxmax()
                    raise InfeasibleSplitError(
                        f'Category {category!r} of column {col!r}: group '
                        f'{offender!r} alone holds {int(worst)} of '
                        f'{int(total)} records. The fold containing that group '
                        f'has only {int(total - worst)} remaining record(s) for '
                        f'its training side but needs at least {m}. Groups '
                        'cannot be split and the threshold cannot be lowered.'
                    )

        # Target labels, canonical order.
        label_series = y.reset_index(drop=True)
        label_values = pd.Index(pd.unique(label_series))
        label_counts_per_group = pd.crosstab(group_series, label_series).reindex(
            group_values, fill_value=0
        )
        label_counts_per_group = label_counts_per_group.reindex(columns=label_values, fill_value=0)

        return {
            'X': X,
            'y': label_series,
            'n_rows': n_rows,
            'group_codes': group_codes,
            'group_values': group_values,
            'group_sizes': group_sizes,
            'category_tables': category_tables,
            'label_table': label_counts_per_group,
            'label_values': label_values,
        }

    def _solve(self, prepared):
        """Build and solve the MILP; return the group-to-fold assignment."""
        k = self.n_splits
        g = len(prepared['group_values'])
        group_sizes = prepared['group_sizes']
        label_table = prepared['label_table'].to_numpy(dtype=float)
        n_labels = label_table.shape[1]
        label_totals = label_table.sum(axis=0)

        # Variables:
        #   x[i, j] in {0, 1}: group i assigned to fold j (i*k + j)
        #   t[j, l] >= 0: abs-deviation helper for fold j, label l
        n_x = g * k
        n_t = k * n_labels
        n_vars = n_x + n_t

        integrality = np.zeros(n_vars)
        integrality[:n_x] = 1
        lower = np.zeros(n_vars)
        upper = np.empty(n_vars)
        upper[:n_x] = 1
        upper[n_x:] = np.inf
        bounds = Bounds(lower, upper)

        constraints = []

        # (1) every group goes to exactly one fold: sum_j x[i,j] = 1
        row_idx, col_idx, vals = [], [], []
        for i in range(g):
            for j in range(k):
                row_idx.append(i)
                col_idx.append(i * k + j)
                vals.append(1.0)
        constraints.append(
            LinearConstraint(
                _sparse_rows(row_idx, col_idx, vals, n_rows=g, n_cols=n_vars),
                np.ones(g),
                np.ones(g),
            )
        )

        # (2) every validation fold non-empty: sum_i size_i x[i,j] >= 1
        row_idx, col_idx, vals = [], [], []
        for j in range(k):
            for i in range(g):
                row_idx.append(j)
                col_idx.append(i * k + j)
                vals.append(group_sizes[i])
        constraints.append(
            LinearConstraint(
                _sparse_rows(row_idx, col_idx, vals, n_rows=k, n_cols=n_vars),
                np.ones(k),
                np.full(k, np.inf),
            )
        )

        # (3) protected categories: training-side count >= min_records.
        # Training side of fold j excludes groups assigned to j, hence
        #   total_c - sum_i count[i, c] x[i, j] >= m.
        cat_rows, cat_lb, cat_ub = [], [], []
        row_offset = 0
        for table in prepared['category_tables'].values():
            counts = table.to_numpy(dtype=float)  # groups x categories
            totals = counts.sum(axis=0)
            for c in range(counts.shape[1]):
                for j in range(k):
                    for i in range(g):
                        if counts[i, c] != 0:
                            cat_rows.append((row_offset, i * k + j, -counts[i, c]))
                    cat_lb.append(self.min_records - totals[c])
                    cat_ub.append(np.inf)
                    row_offset += 1
        if cat_rows:
            row_idx = [r for r, _, _ in cat_rows]
            col_idx = [c for _, c, _ in cat_rows]
            vals = [v for _, _, v in cat_rows]
            constraints.append(
                LinearConstraint(
                    _sparse_rows(row_idx, col_idx, vals, n_rows=row_offset, n_cols=n_vars),
                    np.asarray(cat_lb),
                    np.asarray(cat_ub),
                )
            )

        # (4) objective linearization. Let
        #   a[j, l] = sum_i label_count[i, l] x[i, j], target q_l / k.
        # t[j, l] >= +(a - q/k) and t[j, l] >= -(a - q/k).
        row_idx, col_idx, vals = [], [], []
        n_abs_rows = 0
        for j in range(k):
            for li in range(n_labels):
                base = n_x + j * n_labels + li
                for i in range(g):
                    coefficient = label_table[i, li]
                    if coefficient != 0:
                        row_idx.append(n_abs_rows)
                        col_idx.append(i * k + j)
                        vals.append(coefficient)
                        row_idx.append(n_abs_rows + 1)
                        col_idx.append(i * k + j)
                        vals.append(-coefficient)
                row_idx.append(n_abs_rows)
                col_idx.append(base)
                vals.append(-1.0)
                row_idx.append(n_abs_rows + 1)
                col_idx.append(base)
                vals.append(-1.0)
                n_abs_rows += 2
        n_half = n_abs_rows // 2
        target = label_totals / k
        abs_lb = np.empty(n_abs_rows)
        abs_ub = np.empty(n_abs_rows)
        for half in range(n_half):
            j, li = divmod(half, n_labels)
            # Row A: a - t <= q/k  ->  t >= a - q/k
            abs_lb[2 * half] = -np.inf
            abs_ub[2 * half] = target[li]
            # Row B: -a - t <= -q/k  ->  t >= q/k - a
            abs_lb[2 * half + 1] = -np.inf
            abs_ub[2 * half + 1] = -target[li]
        constraints.append(
            LinearConstraint(
                _sparse_rows(row_idx, col_idx, vals, n_rows=n_abs_rows, n_cols=n_vars),
                abs_lb,
                abs_ub,
            )
        )

        # Single-solve lexicographic objective with normalized, well-conditioned
        # coefficients (a huge big-M weight would degrade the solver's numerical
        # behavior on large instances):
        #
        #   minimize  k * P + S / (S_max + 1)
        #
        # where P = sum of the t variables is the primary label-balance
        # deviation and S = sum_{i,j} i*j*x[i,j] is the deterministic
        # group-order tie-break. Because every feasible integer solution has
        # P on the 1/k grid, k*P is an integer, so two solutions with a
        # different primary value differ by at least 1 on the primary term,
        # while the whole normalized secondary term stays strictly below 1
        # (0 <= S <= S_max). The primary objective therefore always dominates;
        # among primary-optimal partitions the smallest S wins, which puts the
        # earliest canonical groups on the earliest folds.
        s_max = (g * (g - 1) // 2) * (k * (k - 1) // 2)
        objective = np.zeros(n_vars)
        objective[n_x:] = float(k)
        for i in range(g):
            for j in range(k):
                objective[i * k + j] = (i * j) / (s_max + 1)

        options = {'presolve': True}
        if self.random_state is not None:
            options['random_seed'] = int(self.random_state)
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        if result.status == 2:
            raise InfeasibleSplitError(
                'No partition satisfies the constraints: the MILP solver '
                'proved infeasibility. At least one protected category cannot '
                f'keep {self.min_records} training record(s) in every fold '
                'without splitting a group. Reduce n_splits or min_records, or '
                'provide more data; the splitter will not relax either rule.'
            )
        if not result.success:
            raise InfeasibleSplitError(
                f'The MILP solver did not return an optimal partition '
                f'(status={result.status}: {result.message}). No feasible '
                'partition could be certified.'
            )

        x_values = result.x[:n_x].reshape(g, k)
        assignment = np.rint(x_values).astype(int)
        # Defensive checks on the returned integer solution.
        if not np.all(assignment.sum(axis=1) == 1):
            raise RuntimeError('Solver returned a group assigned to != 1 fold.')
        return assignment, result

    def split(self, X, y=None, groups=None):
        """Compute the constrained partition and yield (train, val) index pairs.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data. Must contain every column named in ``cols`` (and in
            ``groups_col`` when groups are taken from X).
        y: array-like, shape (n_samples,)
            Target labels used by the balance objective.
        groups: array-like of shape (n_samples,) or str
            Group identifier per row. A string names a column of ``X``. Rows
            sharing a group always land in the same validation fold. Group
            labels must not be missing.

        Yields
        ------
        train: numpy.ndarray
            Positional indices of the fold's training rows.
        val: numpy.ndarray
            Positional indices of the fold's validation rows.
        """
        prepared = self._prepare(X, y, groups)
        assignment, _ = self._solve(prepared)

        k = self.n_splits
        group_fold = assignment.argmax(axis=1)
        group_codes = prepared['group_codes']
        fold_assignment = group_fold[group_codes]

        folds = [np.flatnonzero(fold_assignment == j) for j in range(k)]
        if any(len(fold) == 0 for fold in folds):
            raise RuntimeError('Solver produced an empty validation fold.')

        # Recompute every statistic directly from the partition rather than
        # from MILP internals, so the reported counts are trustworthy.
        coverage = self._build_coverage(prepared, folds)
        primary_value = self._primary_objective(prepared, coverage)

        self.result_ = ConstrainedSplitResult(
            folds=folds,
            fold_assignment=fold_assignment,
            groups=list(prepared['group_values']),
            group_folds={
                group: int(fold)
                for group, fold in zip(prepared['group_values'], group_fold, strict=True)
            },
            coverage=coverage,
            objective_value=primary_value,
            cols=list(self._cols_list),
            min_records=self.min_records,
        )

        n_rows = prepared['n_rows']
        all_indices = np.arange(n_rows)
        for val_idx in folds:
            mask = np.ones(n_rows, dtype=bool)
            mask[val_idx] = False
            yield all_indices[mask], val_idx

    @staticmethod
    def _primary_objective(prepared, coverage):
        """Sum |fold label count - total label count / K| from real counts."""
        label_totals = prepared['label_table'].sum(axis=0)
        k = len(coverage)
        total = 0.0
        for label in prepared['label_values']:
            target = float(label_totals[label]) / k
            for cov in coverage:
                total += abs(cov.val_label_counts.get(label, 0) - target)
        return total

    def _build_coverage(self, prepared, folds):
        """Build FoldCoverage statistics from an assignment."""
        X = prepared['X']
        y_values = prepared['y'].to_numpy()
        n_rows = prepared['n_rows']
        coverage = []
        for j, val_idx in enumerate(folds):
            val_mask = np.zeros(n_rows, dtype=bool)
            val_mask[val_idx] = True
            train_mask = ~val_mask
            train_category_counts = {}
            min_train = {}
            for col in self._cols_list:
                counts = X.loc[train_mask, col].value_counts(dropna=False).to_dict()
                train_category_counts[col] = counts
                min_train[col] = min(counts.values())
            val_label_counts = pd.Series(y_values[val_mask]).value_counts().to_dict()
            coverage.append(
                FoldCoverage(
                    fold_id=j,
                    n_val=int(val_mask.sum()),
                    n_train=int(train_mask.sum()),
                    val_label_counts={
                        label: int(count) for label, count in val_label_counts.items()
                    },
                    train_category_counts={
                        col: {category: int(count) for category, count in per_category.items()}
                        for col, per_category in train_category_counts.items()
                    },
                    min_train_category_counts={
                        col: int(minimum) for col, minimum in min_train.items()
                    },
                )
            )
        return coverage

    def fit(self, X, y=None, groups=None):
        """Compute the partition and store the result on ``self.result_``."""
        for _ in self.split(X, y, groups):
            pass
        return self

    def __repr__(self) -> str:
        """Unambiguous constructor-style representation."""
        return (
            f'ConstrainedGroupKFold(n_splits={self.n_splits}, cols={self.cols!r}, '
            f'min_records={self.min_records}, groups_col={self.groups_col!r}, '
            f'random_state={self.random_state!r})'
        )


def _sparse_rows(row_idx, col_idx, values, n_rows, n_cols):
    """Build a CSR matrix from COO triples (scipy sparse avoids dense MILP rows)."""
    return csr_matrix(
        (np.asarray(values, dtype=float), (np.asarray(row_idx), np.asarray(col_idx))),
        shape=(n_rows, n_cols),
    )


def split_constrained_group_kfold(X, y, groups, n_splits, cols, min_records=1, random_state=None):
    """Functional entry returning folds together with coverage statistics.

    This is a thin convenience wrapper around :class:`ConstrainedGroupKFold`
    for callers that want the :class:`ConstrainedSplitResult` directly without
    driving the sklearn ``split`` iterator themselves.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Data containing every column named in ``cols``.
    y: array-like, shape (n_samples,)
        Target labels used by the balance objective.
    groups: array-like of shape (n_samples,) or str
        Group identifier per row (a string names a column of ``X``).
    n_splits: int
        Number of non-empty validation folds K.
    cols: str or list of str
        Columns whose categories must retain ``min_records`` training rows in
        every fold.
    min_records: int, default=1
        Minimum training-side count per protected category per fold.
    random_state: int, optional
        Seed for the deterministic HiGHS solver.

    Returns
    -------
    ConstrainedSplitResult
        The folds, the row-level fold assignment, group mapping and coverage
        statistics.

    Raises
    ------
    InfeasibleSplitError
        If no partition meets the requirements, with a reason in the message.
    """
    splitter = ConstrainedGroupKFold(
        n_splits=n_splits,
        cols=cols,
        min_records=min_records,
        random_state=random_state,
    )
    for _ in splitter.split(X, y, groups):
        pass
    return splitter.result_
