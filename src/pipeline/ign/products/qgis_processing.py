from __future__ import annotations

from typing import Any, Dict, Optional


def run_qgis_algorithm(
    algorithm_id: str,
    parameters: Dict[str, Any],
    *,
    feedback: Optional[Any] = None,
    context: Optional[Any] = None,
) -> Dict[str, Any]:
    try:
        import processing  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Le module 'processing' est indisponible. Cette étape doit être exécutée depuis QGIS (plugin)."
        ) from e

    kwargs: Dict[str, Any] = {}
    if feedback is not None:
        kwargs["feedback"] = feedback
    if context is not None:
        kwargs["context"] = context

    return processing.run(algorithm_id, parameters, **kwargs)
