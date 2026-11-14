BACKENDS = ("auto", "sklearn", "cuml")


class BackendUnavailable(RuntimeError):
    pass


def _check(backend):
    if backend not in BACKENDS:
        raise ValueError(
            "unknown backend %r; valid backends: %s" % (backend, list(BACKENDS))
        )


def _asarray(X):
    import numpy as np

    if hasattr(X, "to_numpy"):
        return np.ascontiguousarray(X.to_numpy())
    return X


_CUML_SVC = None
_CUML_BASE = None


def _cuml_svc():
    global _CUML_SVC, _CUML_BASE
    from cuml.svm import SVC

    if _CUML_SVC is None or _CUML_BASE is not SVC:

        class SVCWithoutProba(SVC):
            @property
            def predict_proba(self):
                raise AttributeError("predict_proba requires probability=True")

            def fit(self, X, y=None, sample_weight=None, **kwargs):
                return super().fit(
                    _asarray(X), y, sample_weight=sample_weight, **kwargs
                )

            def predict(self, X, **kwargs):
                return super().predict(_asarray(X), **kwargs)

            def decision_function(self, X, **kwargs):
                return super().decision_function(_asarray(X), **kwargs)

        SVCWithoutProba.__module__ = __name__
        SVCWithoutProba.__qualname__ = "SVCWithoutProba"
        _CUML_BASE = SVC
        _CUML_SVC = SVCWithoutProba
    return _CUML_SVC


def __getattr__(name):
    if name == "SVCWithoutProba":
        return _cuml_svc()
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def resolve_svc(backend="auto"):
    _check(backend)
    if backend in ("auto", "cuml"):
        try:
            return "cuml", _cuml_svc()
        except Exception as error:
            if backend == "cuml":
                raise BackendUnavailable(
                    "the cuml backend was requested but cuml could not be "
                    "imported (%s); cuml ships for linux only, so on Windows run "
                    "under WSL2 or pass svc_backend='sklearn'" % error
                )
    from sklearn.svm import SVC

    return "sklearn", SVC


def build_svc(seed, *, class_weight, backend="auto", kernel="rbf"):
    name, cls = resolve_svc(backend)
    if name == "cuml":
        return name, cls(kernel=kernel, class_weight=class_weight, random_state=seed)
    return name, cls(
        kernel=kernel,
        class_weight=class_weight,
        probability=False,
        random_state=seed,
    )


def limit_threads(n_threads):
    """Return a context manager capping BLAS and OpenMP threads, or a no-op
    context manager when threadpoolctl is unavailable."""
    import contextlib

    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return contextlib.nullcontext()
    return threadpool_limits(limits=int(n_threads))

