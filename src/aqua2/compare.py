"""Strict recursive parity reports. Default float policy is exact equality."""

from dataclasses import dataclass, asdict

import numpy as np

from .io import MatStructArray


@dataclass(frozen=True)
class Tolerance:
    atol: float = 0.0
    rtol: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.atol) or not np.isfinite(self.rtol) or self.atol < 0 or self.rtol < 0:
            raise ValueError("Tolerances must be finite and nonnegative")


def compare(reference, actual, *, tolerance=Tolerance(), check_dtype=True):
    issues = []
    arrays_checked = 0

    def fail(path, reason, **details):
        issues.append({"field": path, "reason": reason, **details})

    def visit(left, right, path):
        nonlocal arrays_checked
        if isinstance(left, MatStructArray) != isinstance(right, MatStructArray):
            fail(path, "matlab_struct_vs_cell")
            return
        if isinstance(left, MatStructArray) and set(left.mat_fields) != set(right.mat_fields):
            fail(path, "matlab_struct_fields")
            return
        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict):
                fail(path, "type")
                return
            for key in sorted(left.keys() | right.keys()):
                if key not in left or key not in right:
                    fail(f"{path}/{key}", "missing", side="reference" if key not in left else "actual")
                else:
                    visit(left[key], right[key], f"{path}/{key}")
            return
        a, b = np.asarray(left), np.asarray(right)
        arrays_checked += 1
        if a.dtype.kind == "c" or b.dtype.kind == "c":
            raise TypeError("Complex parity is not supported; compare real and imaginary fields explicitly")
        if a.shape != b.shape:
            fail(path, "shape", reference=list(a.shape), actual=list(b.shape))
            return
        if a.dtype == object or b.dtype == object:
            if a.dtype != b.dtype:
                fail(path, "type")
                return
            for index in np.ndindex(a.shape):
                visit(a[index], b[index], path + str(index))
            return
        if check_dtype and a.dtype != b.dtype:
            fail(path, "dtype", reference=str(a.dtype), actual=str(b.dtype))
        if a.dtype.kind in "biufc" and b.dtype.kind in "biufc":
            same_special = (np.array_equal(np.isnan(a), np.isnan(b)) and
                            np.array_equal(np.isposinf(a), np.isposinf(b)) and
                            np.array_equal(np.isneginf(a), np.isneginf(b)))
            if not same_special:
                fail(path, "nan_or_inf_distribution")
            finite = np.isfinite(a) & np.isfinite(b)
            af, bf = a[finite], b[finite]
            # Discrete values never receive floating tolerance, even on dtype mismatch.
            discrete = a.dtype.kind in "biu" or b.dtype.kind in "biu"
            equal = (af == bf) if discrete else np.less_equal(
                np.abs(af.astype(np.float64) - bf.astype(np.float64)),
                tolerance.atol + tolerance.rtol * np.abs(af.astype(np.float64)),
            )
            if not np.all(equal):
                difference = np.abs(af.astype(np.float64) - bf.astype(np.float64))
                mismatch = np.zeros(a.shape, dtype=bool)
                mismatch[finite] = ~equal
                first = np.argwhere(mismatch)[0].tolist()
                fail(path, "values", mismatch_count=int(np.sum(~equal)),
                     max_abs_error=float(np.max(difference)), first_index=first)
        elif not np.array_equal(a, b):
            fail(path, "values")

    visit(reference, actual, "root")
    return {"passed": not issues, "tolerance": asdict(tolerance),
            "check_dtype": check_dtype, "arrays_checked": arrays_checked,
            "first_difference": issues[0] if issues else None, "issues": issues}


def compare_stages(reference, actual, **kwargs):
    """Insertion order of the reference dictionary defines pipeline order."""
    reports = {name: compare(value, actual.get(name), **kwargs) for name, value in reference.items()}
    extras = sorted(actual.keys() - reference.keys())
    return {"passed": all(r["passed"] for r in reports.values()) and not extras,
            "first_differing_stage": next((name for name, report in reports.items() if not report["passed"]), None),
            "unexpected_stages": extras, "stages": reports}
