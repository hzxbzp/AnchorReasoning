"""Native-pipeline runners for released models that are not driven through a shared adapter:
Alpamayo-R1 / 1.5, AutoVLA and Impromptu-VLA.

Each runner executes the model's own inference code (its own virtual environment, prompt, sampling
and action head) on the rated validation frames and writes rows in the ``chains.json`` schema used by
``evaluation.eval_val456``, so ``evaluation.baselines.rescore`` can score them exactly like the
adapter-driven zero-shot runs:
    scene_id, frame_id, fdir, task, text, pred_xy (20 x 2 @ 0.25 s), n_new_tokens, hit_cap,
    prompt_len, native {...}
The runners import ``common.py`` by path so they work from any of the model environments.
"""
