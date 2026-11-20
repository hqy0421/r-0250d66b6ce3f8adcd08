"""Tests for the constrained customer-aware K-fold splitter.

The splitter must:
* keep every group (customer) inside one validation fold,
* keep every fold non-empty,
* guarantee min_records training-side rows per protected category,
* globally minimize the per-fold, per-label absolute deviation objective,
* break ties deterministically,
* report a concrete reason when no partition exists.
"""

from itertools import product
from unittest import TestCase, mock

import category_encoders as encoders
import numpy as np
import pandas as pd
from category_encoders import cv_split as cv_split_module
from category_encoders.cv_split import (
    ConstrainedGroupKFold,
    ConstrainedSplitResult,
    FoldCoverage,
    InfeasibleSplitError,
    split_constrained_group_kfold,
)
from category_encoders.wrapper import NestedCVWrapper
from scipy.optimize import OptimizeResult
from sklearn.base import clone
from sklearn.model_selection import check_cv

# 6 customers, 2 rows each. Every category spans at least 4 customers so a
# 3-fold split with min_records=1 is feasible.
X_BASE = pd.DataFrame(
    {
        'cat': ['a', 'b', 'b', 'a', 'c', 'a', 'c', 'b', 'a', 'c', 'b', 'c'],
        'num': np.arange(12),
    }
)
Y_BASE = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
GROUPS_BASE = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])


def _brute_force_best(X, y, groups, n_splits, min_records, cols):
    """Enumerate every group->fold assignment and return the best objective.

    Returns (best_objective, n_feasible). Feasibility mirrors the splitter:
    non-empty folds and min_records training rows for every category.
    """
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    y = np.asarray(y)
    label_totals = pd.Series(y).value_counts()
    best = np.inf
    n_feasible = 0
    for assignment in product(range(n_splits), repeat=len(unique_groups)):
        if len(set(assignment)) < n_splits:
            continue  # an empty validation fold
        group_to_fold = dict(zip(unique_groups, assignment, strict=True))
        row_fold = np.array([group_to_fold[g] for g in groups])
        feasible = True
        for fold_id in range(n_splits):
            train_mask = row_fold != fold_id
            for col in cols:
                counts = X.loc[train_mask, col].value_counts(dropna=False)
                if len(counts) != X[col].nunique(dropna=False) or counts.min() < min_records:
                    feasible = False
                    break
            if not feasible:
                break
        if not feasible:
            continue
        n_feasible += 1
        objective = 0.0
        for label, total in label_totals.items():
            target = total / n_splits
            for fold_id in range(n_splits):
                observed = int(((row_fold == fold_id) & (y == label)).sum())
                objective += abs(observed - target)
        best = min(best, objective)
    return best, n_feasible


class TestConstrainedGroupKFoldBasics(TestCase):
    """Structural properties of a feasible partition."""

    def test_groups_never_split_and_folds_partition_rows(self):
        """Each customer lands in one fold; folds partition all rows."""
        splitter = ConstrainedGroupKFold(n_splits=3, cols=['cat'], min_records=1)
        splits = list(splitter.split(X_BASE, Y_BASE, GROUPS_BASE))
        result = splitter.result_

        self.assertEqual(3, len(splits))
        # every fold non-empty
        self.assertTrue(all(len(val) > 0 for _, val in splits))
        # train and val are disjoint and cover every row exactly once
        val_concat = np.sort(np.concatenate([val for _, val in splits]))
        np.testing.assert_array_equal(val_concat, np.arange(len(X_BASE)))
        for trn, val in splits:
            self.assertFalse(set(trn) & set(val))
            self.assertEqual(len(trn) + len(val), len(X_BASE))

        # group integrity at the row level
        for group in np.unique(GROUPS_BASE):
            member_folds = set(result.fold_assignment[GROUPS_BASE == group])
            self.assertEqual(1, len(member_folds))
        # group_folds agrees with the row-level assignment
        for group, fold_id in result.group_folds.items():
            member_rows = np.where(GROUPS_BASE == group)[0]
            self.assertTrue((result.fold_assignment[member_rows] == fold_id).all())

    def test_min_records_holds_on_every_training_side(self):
        """Every protected category keeps min_records rows in each train side."""
        splitter = ConstrainedGroupKFold(n_splits=2, cols=['cat'], min_records=2)
        splits = list(splitter.split(X_BASE, Y_BASE, GROUPS_BASE))
        for trn, _ in splits:
            counts = X_BASE.iloc[trn]['cat'].value_counts()
            self.assertGreaterEqual(counts.min(), 2)
        for coverage in splitter.result_.coverage:
            self.assertTrue(
                all(minimum >= 2 for minimum in coverage.min_train_category_counts.values())
            )

    def test_multiple_protected_columns(self):
        """Constraints apply jointly to every column in cols."""
        X = X_BASE.copy()
        X['other'] = ['x', 'y'] * 6
        splitter = ConstrainedGroupKFold(n_splits=2, cols=['cat', 'other'], min_records=1)
        splits = list(splitter.split(X, Y_BASE, GROUPS_BASE))
        for trn, _val in splits:
            self.assertGreaterEqual(X.iloc[trn]['cat'].value_counts().min(), 1)
            self.assertGreaterEqual(X.iloc[trn]['other'].value_counts().min(), 1)
        self.assertEqual({'cat', 'other'}, set(splitter.result_.coverage[0].train_category_counts))

    def test_nan_is_a_protected_category(self):
        """Missing values form an explicit category and get their own counts."""
        X = pd.DataFrame({'cat': ['a', np.nan, 'b', 'a', np.nan, 'b', 'a', np.nan]})
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        groups = [1, 1, 2, 2, 3, 3, 4, 4]
        splitter = ConstrainedGroupKFold(n_splits=2, cols=['cat'], min_records=1)
        result = splitter.fit(X, y, groups).result_

        for coverage in result.coverage:
            counts = coverage.train_category_counts['cat']
            self.assertIn(np.nan, counts)
            self.assertGreaterEqual(counts[np.nan], 1)

    def test_functional_entry_returns_full_result(self):
        """The functional entry returns folds, assignment and statistics."""
        result = split_constrained_group_kfold(
            X_BASE, Y_BASE, GROUPS_BASE, n_splits=3, cols='cat', min_records=1
        )
        self.assertIsInstance(result, ConstrainedSplitResult)
        self.assertEqual(3, result.n_splits)
        self.assertEqual(3, len(result.coverage))
        self.assertIsInstance(result.coverage[0], FoldCoverage)
        # summary and long-form coverage frames are populated
        summary = result.summary_frame()
        self.assertEqual(3, len(summary))
        self.assertIn('min_train_count:cat', summary.columns)
        coverage = result.coverage_frame()
        self.assertIn('__label__', set(coverage['col']))

    def test_groups_column_name_and_groups_col_constructor(self):
        """A string group argument can name an X column."""
        X = X_BASE.copy()
        X['customer'] = GROUPS_BASE
        splitter = ConstrainedGroupKFold(n_splits=3, cols=['cat'], groups_col='customer')
        result = splitter.fit(X, Y_BASE).result_
        self.assertEqual(3, result.n_splits)

        result2 = split_constrained_group_kfold(
            X, Y_BASE, groups='customer', n_splits=3, cols=['cat']
        )
        np.testing.assert_array_equal(result.fold_assignment, result2.fold_assignment)

    def test_sklearn_cv_interface(self):
        """The splitter is a valid sklearn cross-validator."""
        splitter = ConstrainedGroupKFold(n_splits=3, cols=['cat'])
        checked = check_cv(splitter, y=Y_BASE, classifier=True)
        self.assertEqual(3, checked.get_n_splits())
        # constructor parameters survive sklearn's clone
        cloned = clone(splitter)
        self.assertEqual(3, cloned.n_splits)
        self.assertEqual(['cat'], cloned.cols)
        self.assertEqual(1, cloned.min_records)
        # a cloned, unfitted splitter produces the same partition
        cloned.fit(X_BASE, Y_BASE, GROUPS_BASE)
        np.testing.assert_array_equal(
            splitter.fit(X_BASE, Y_BASE, GROUPS_BASE).result_.fold_assignment,
            cloned.result_.fold_assignment,
        )


class TestOptimalityAndStability(TestCase):
    """The MILP objective is globally optimal and deterministic."""

    def test_matches_brute_force_optimum_on_small_cases(self):
        """Enumerate all assignments; the MILP must hit the same best objective."""
        rng = np.random.default_rng(33)
        for _case in range(6):
            n_groups = 5
            groups = np.repeat(np.arange(n_groups), 2)
            X = pd.DataFrame({'cat': rng.choice(['a', 'b', 'c'], size=2 * n_groups)})
            y = pd.Series(rng.integers(0, 2, size=2 * n_groups))
            n_splits = 2
            min_records = 1
            best, n_feasible = _brute_force_best(X, y, groups, n_splits, min_records, ['cat'])
            splitter = ConstrainedGroupKFold(n_splits, ['cat'], min_records)
            if n_feasible == 0:
                with self.assertRaises(InfeasibleSplitError):
                    splitter.fit(X, y, groups)
                continue
            result = splitter.fit(X, y, groups).result_
            self.assertAlmostEqual(best, result.objective_value)
            # recomputed stats agree with the reported objective
            recomputed = splitter._primary_objective(
                splitter._prepare(X, y, groups), result.coverage
            )
            self.assertAlmostEqual(result.objective_value, recomputed)

    def test_four_fold_primary_objective_matches_brute_force(self):
        """The primary objective remains globally optimal for K >= 4.

        Brute force is the reference; this guards both the absolute-value
        linearization and the lexicographic (primary-first) optimization.
        """
        rng = np.random.default_rng(91)
        n_groups = 6
        groups = np.repeat(np.arange(n_groups), 2)
        n_splits = 4
        checked = 0
        for _case in range(10):
            X = pd.DataFrame({'cat': rng.choice(['a', 'b'], size=2 * n_groups)})
            y = pd.Series(rng.integers(0, 3, size=2 * n_groups))
            best, n_feasible = _brute_force_best(X, y, groups, n_splits, 1, ['cat'])
            splitter = ConstrainedGroupKFold(n_splits, ['cat'], min_records=1)
            if n_feasible == 0:
                with self.assertRaises(InfeasibleSplitError):
                    splitter.fit(X, y, groups)
                continue
            result = splitter.fit(X, y, groups).result_
            self.assertAlmostEqual(best, result.objective_value)
            checked += 1
        self.assertGreater(checked, 0)

    def test_imbalanced_labels_still_feasible(self):
        """Skewed label counts must not make a feasible problem look infeasible.

        Regression test for the absolute-value linearization: the second
        half-space is an upper bound, so a label whose total is not divisible
        by K does not tighten into an impossible constraint.
        """
        # 5 groups of 2 rows; only two rows carry the positive label.
        X = pd.DataFrame(
            {
                'cat': ['a', 'b', 'b', 'a', 'c', 'a', 'c', 'b', 'a', 'c'],
            }
        )
        y = pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        groups = np.repeat(np.arange(5), 2)
        result = ConstrainedGroupKFold(2, ['cat'], min_records=1).fit(X, y, groups).result_
        best, n_feasible = _brute_force_best(X, y, groups, 2, 1, ['cat'])
        self.assertGreater(n_feasible, 0)
        self.assertAlmostEqual(best, result.objective_value)

    def test_repeated_calls_are_identical(self):
        """Same data, separate instances -> identical partition."""
        first = (
            ConstrainedGroupKFold(n_splits=3, cols=['cat']).fit(X_BASE, Y_BASE, GROUPS_BASE).result_
        )
        second = (
            ConstrainedGroupKFold(n_splits=3, cols=['cat']).fit(X_BASE, Y_BASE, GROUPS_BASE).result_
        )
        np.testing.assert_array_equal(first.fold_assignment, second.fold_assignment)
        self.assertEqual(first.group_folds, second.group_folds)

    def test_row_order_does_not_change_partition_semantics(self):
        """Shuffling rows yields the same group->fold mapping."""
        permutation = np.random.default_rng(1).permutation(len(X_BASE))
        X_shuffled = X_BASE.iloc[permutation].reset_index(drop=True)
        y_shuffled = Y_BASE.iloc[permutation].reset_index(drop=True)
        groups_shuffled = GROUPS_BASE[permutation]

        regular = (
            ConstrainedGroupKFold(n_splits=3, cols=['cat']).fit(X_BASE, Y_BASE, GROUPS_BASE).result_
        )
        shuffled = (
            ConstrainedGroupKFold(n_splits=3, cols=['cat'])
            .fit(X_shuffled, y_shuffled, groups_shuffled)
            .result_
        )
        self.assertEqual(regular.group_folds, shuffled.group_folds)

    def test_tie_break_is_deterministic_and_canonical(self):
        """With a perfectly symmetric dataset the same partition always wins.

        Every 2+2 partition has objective 0 here, so only the deterministic
        secondary group-order criterion decides. The winner must be identical
        across runs and assign folds in a contiguous canonical order.
        """
        X = pd.DataFrame({'cat': ['a', 'a', 'a', 'a']})
        y = pd.Series([0, 1, 0, 1])
        groups = [0, 1, 2, 3]
        first = ConstrainedGroupKFold(n_splits=2, cols=['cat']).fit(X, y, groups).result_
        second = ConstrainedGroupKFold(n_splits=2, cols=['cat']).fit(X, y, groups).result_
        self.assertEqual(first.group_folds, second.group_folds)
        self.assertEqual(0.0, first.objective_value)
        # the canonical winner is a contiguous split of the sorted groups
        assigned = [first.group_folds[g] for g in range(4)]
        self.assertEqual([1, 1, 0, 0], assigned)


class TestInfeasibility(TestCase):
    """No-solution cases raise with an explanatory message and no relaxation."""

    def test_category_total_below_requirement(self):
        """A category concentrated in one group cannot meet the training floor."""
        # Both 'a' rows sit in group 1; the fold hosting it keeps zero 'a'
        # training rows but min_records=2 demands two.
        X = pd.DataFrame({'cat': ['a', 'a', 'b', 'b', 'b', 'b']})
        y = pd.Series([0, 1, 0, 1, 0, 1])
        groups = np.array([1, 1, 2, 2, 3, 3])
        with self.assertRaisesRegex(InfeasibleSplitError, 'alone holds 2 of 2 records'):
            list(ConstrainedGroupKFold(2, ['cat'], min_records=2).split(X, y, groups))

    def test_sparse_but_distributed_category_is_feasible(self):
        """Few total rows still work when spread across enough groups.

        Guards against an over-eager total-count rejection: K=3, m=2 with
        only 3 'b' rows, one per group, is feasible because no single fold can
        exclude more than one of them.
        """
        # groups g1..g4, two rows each; 'b' appears once in g1, g2, g3.
        X = pd.DataFrame({'cat': ['b', 'a', 'b', 'a', 'b', 'a', 'a', 'a']})
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        result = ConstrainedGroupKFold(3, ['cat'], min_records=2).fit(X, y, groups).result_
        for coverage in result.coverage:
            self.assertGreaterEqual(coverage.train_category_counts['cat'].get('b', 0), 2)

    def test_group_concentration_blocks_partition(self):
        """One group holding most of a category cannot be neutralized."""
        # K=2, m=2: 4 'a' rows total but one group holds 3 of them, so the fold
        # hosting that group has only one 'a' row left for its training side.
        X = pd.DataFrame({'cat': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']})
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        groups = np.array([1, 1, 1, 2, 2, 3, 3, 4])
        with self.assertRaisesRegex(InfeasibleSplitError, 'alone holds'):
            list(ConstrainedGroupKFold(2, ['cat'], min_records=2).split(X, y, groups))

    def test_fewer_groups_than_folds(self):
        """Groups cannot be split to populate extra folds."""
        groups = [1, 1, 1, 2, 2, 2]
        with self.assertRaisesRegex(InfeasibleSplitError, 'Only 2 group'):
            list(ConstrainedGroupKFold(3, ['cat']).split(X_BASE.iloc[:6], Y_BASE.iloc[:6], groups))

    def test_milp_level_infeasibility_without_relaxation(self):
        """If HiGHS itself proves infeasibility, the reason is reported."""
        splitter = ConstrainedGroupKFold(n_splits=3, cols=['cat'], min_records=1)
        with mock.patch.object(
            cv_split_module,
            'milp',
            return_value=OptimizeResult(status=2, success=False, message='infeasible', x=None),
        ):
            with self.assertRaisesRegex(InfeasibleSplitError, 'proved infeasibility'):
                list(splitter.split(X_BASE, Y_BASE, GROUPS_BASE))
        self.assertIsNone(splitter.result_)

    def test_solver_failure_is_reported_not_returned(self):
        """A non-optimal solver exit raises instead of yielding a bad split."""
        splitter = ConstrainedGroupKFold(n_splits=3, cols=['cat'], min_records=1)
        with mock.patch.object(
            cv_split_module,
            'milp',
            return_value=OptimizeResult(
                status=4, success=False, message='time limit reached', x=None
            ),
        ):
            with self.assertRaisesRegex(InfeasibleSplitError, 'did not return an optimal'):
                list(splitter.split(X_BASE, Y_BASE, GROUPS_BASE))
        self.assertIsNone(splitter.result_)

    def test_invalid_arguments(self):
        """Bad constructor/call arguments raise ordinary ValueErrors."""
        with self.assertRaises(ValueError):
            ConstrainedGroupKFold(1, ['cat'])
        with self.assertRaises(ValueError):
            ConstrainedGroupKFold(3, [])
        with self.assertRaises(TypeError):
            ConstrainedGroupKFold(3, ['cat'], min_records=1.5)
        with self.assertRaises(ValueError):
            ConstrainedGroupKFold(3, ['cat'], min_records=0)
        with self.assertRaises(ValueError):
            list(
                ConstrainedGroupKFold(3, ['cat']).split(
                    X_BASE, Y_BASE, groups=[1, 1, None, 2, 2, 2, 3, 3, 3, 4, 4, 4]
                )
            )
        with self.assertRaises(ValueError):
            list(ConstrainedGroupKFold(3, ['nope']).split(X_BASE, Y_BASE, GROUPS_BASE))
        with self.assertRaises(ValueError):
            list(ConstrainedGroupKFold(3, ['cat']).split(X_BASE, None, GROUPS_BASE))


class TestNestedCVWrapperIntegration(TestCase):
    """The native wrapper drives out-of-fold encoding from the real split."""

    def test_wrapper_uses_constrained_folds_and_preserves_order(self):
        """Encoding folds equal the splitter's; rows return in input order."""
        cv = ConstrainedGroupKFold(n_splits=3, cols=['cat'], min_records=1)
        wrapper = NestedCVWrapper(encoders.MEstimateEncoder(m=1, cols=['cat']), cv=cv)
        out = wrapper.fit_transform(X_BASE, Y_BASE, groups=GROUPS_BASE)

        np.testing.assert_array_equal(out.index.values, X_BASE.index.values)
        np.testing.assert_array_equal(out['num'].to_numpy(), X_BASE['num'].to_numpy())
        # the wrapper records the exact folds and the constrained result
        self.assertEqual(3, len(wrapper.cv_split_))
        self.assertIs(cv.result_, wrapper.cv_result_)
        for (trn, val), (cv_trn, cv_val) in zip(
            wrapper.cv_split_, cv.split(X_BASE, Y_BASE, GROUPS_BASE), strict=True
        ):
            np.testing.assert_array_equal(np.sort(trn), np.sort(cv_trn))
            np.testing.assert_array_equal(np.sort(val), np.sort(cv_val))
        # customers never cross the train/val boundary during encoding
        for trn, val in wrapper.cv_split_:
            self.assertFalse(set(GROUPS_BASE[trn]) & set(GROUPS_BASE[val]))

    def test_oof_encoding_matches_manual_fold_encoding(self):
        """Each out-of-fold row equals an encoder fit on its fold's train side."""
        cv = ConstrainedGroupKFold(n_splits=3, cols=['cat'], min_records=1)
        wrapper = NestedCVWrapper(encoders.TargetEncoder(cols=['cat']), cv=cv)
        out = wrapper.fit_transform(X_BASE, Y_BASE, groups=GROUPS_BASE)

        for trn, val in wrapper.cv_split_:
            manual = encoders.TargetEncoder(cols=['cat'])
            manual.fit(X_BASE.iloc[trn], Y_BASE.iloc[trn])
            expected = manual.transform(X_BASE.iloc[val])
            np.testing.assert_allclose(out.iloc[val]['cat'].to_numpy(), expected['cat'].to_numpy())

    def test_wrapper_with_group_column_name(self):
        """A string groups argument resolves against X's columns."""
        X = X_BASE.copy()
        X['customer'] = GROUPS_BASE
        cv = ConstrainedGroupKFold(n_splits=3, cols=['cat'], groups_col='customer')
        wrapper = NestedCVWrapper(encoders.MEstimateEncoder(m=1, cols=['cat']), cv=cv)
        out = wrapper.fit_transform(X, Y_BASE)
        np.testing.assert_array_equal(out['num'].to_numpy(), X['num'].to_numpy())
        self.assertIsNotNone(wrapper.cv_result_)

    def test_infeasible_split_propagates_from_wrapper(self):
        """The wrapper does not paper over an infeasible constrained split."""
        X = pd.DataFrame({'cat': ['a', 'a', 'b', 'b', 'b', 'b']})
        y = pd.Series([0, 1, 0, 1, 0, 1])
        groups = np.array([1, 1, 2, 2, 3, 3])
        cv = ConstrainedGroupKFold(n_splits=2, cols=['cat'], min_records=2)
        wrapper = NestedCVWrapper(encoders.TargetEncoder(cols=['cat']), cv=cv)
        with self.assertRaises(InfeasibleSplitError):
            wrapper.fit_transform(X, y, groups=groups)

    def test_plain_sklearn_splitter_still_records_folds(self):
        """cv_split_ is populated for ordinary splitters too; cv_result_ is None."""
        from sklearn.model_selection import KFold

        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        wrapper = NestedCVWrapper(encoders.MEstimateEncoder(m=1, cols=['cat']), cv=cv)
        out = wrapper.fit_transform(X_BASE, Y_BASE)
        self.assertEqual(3, len(wrapper.cv_split_))
        self.assertIsNone(wrapper.cv_result_)
        np.testing.assert_array_equal(out['num'].to_numpy(), X_BASE['num'].to_numpy())
