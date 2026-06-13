# AORA-Forge — captured demo transcripts

Output of the two demos run in **mock mode** (no API key) on the synthetic
failure set, captured during the overnight build. Reproduce with
`make demo` and `make hook-demo` (set `ANTHROPIC_API_KEY` for the real-API run).

## `python scripts/demo_full_loop.py --mock`

```

==============================================================================
  AORA-Forge — end-to-end growth demo
==============================================================================

Generated 50 synthetic failures -> /tmp/transcript_demo/synthetic_failures.jsonl
  by embodiment : {'drone_figs': 27, 'ground_habitat': 23}
  failure modes : ['CLAIMED_NOT_REACHED', 'DONE_GATE_LOOP', 'HALLUCINATED_DONE', 'INSTANCE_CONFUSION', 'OOB', 'SCAN_LOOP', 'STUCK_AGAINST_WALL', 'TARGET_MISIDENTIFICATION', 'TARGET_NOT_VISIBLE', 'TIMEOUT_NO_PROGRESS', 'WRONG_ROOM']

==============================================================================
  Growing skills from failures
==============================================================================

==============================================================================
  Forged skills
==============================================================================
9 clusters -> 9 skills (9 validated)

  [PASS] premature_success_on_small_or_cluttered_targets_prompt
         type=prompt     score=0.85  cases=6/6 embodiments=[drone_figs,ground_habitat]
  [PASS] target_disambiguation_among_look_alikes_classifier
         type=classifier score=1.0   cases=80/80 embodiments=[drone_figs,ground_habitat]
         3DGS substrate: recon_1f12d842b3 (stub)
  [PASS] arrival_gate_thrashing_near_the_target_code
         type=code       score=1.0   cases=2/2 embodiments=[drone_figs,ground_habitat]
  [PASS] boundary_geofence_handling_code
         type=code       score=1.0   cases=2/2 embodiments=[drone_figs]
  [PASS] collision_recovery_in_clutter_code
         type=code       score=1.0   cases=2/2 embodiments=[ground_habitat]
  [PASS] search_under_occlusion_out_of_view_targets_prompt
         type=prompt     score=0.85  cases=5/5 embodiments=[drone_figs,ground_habitat]
  [PASS] long_horizon_exploration_stalls_prompt
         type=prompt     score=0.85  cases=5/5 embodiments=[drone_figs,ground_habitat]
  [PASS] unproductive_scanning_without_progress_code
         type=code       score=1.0   cases=2/2 embodiments=[drone_figs,ground_habitat]
  [PASS] multi_room_goal_localisation_prompt
         type=prompt     score=0.85  cases=3/3 embodiments=[drone_figs,ground_habitat]

==============================================================================
  Scene-conditioned retrieval (skills -> planner tools)
==============================================================================

  scene ['green clock', 'table'] on drone_figs:
    -> target_disambiguation_among_look_alikes_classifier  [direct_tool]
    -> premature_success_on_small_or_cluttered_targets_prompt  [executor_prompt]

  scene ['chair', 'sofa'] on ground_habitat:
    -> collision_recovery_in_clutter_code  [direct_tool]
    -> unproductive_scanning_without_progress_code  [direct_tool]
    -> target_disambiguation_among_look_alikes_classifier  [direct_tool]

  scene ['potted plant'] on ground_habitat:
    -> arrival_gate_thrashing_near_the_target_code  [direct_tool]

==============================================================================
  Run summary — what this cost
==============================================================================
  LLM backend          : MOCK (no API key)
  Wall-clock           :    0.1 s
  Total input tokens   : 0
    cache read         : 0
    cache creation     : 0
  Total output tokens  : 0
  Estimated cost (USD) : $0.0000
  Calls by stage:
    cluster_failures               1 call(s)  $0.0000
    generate_spec                  9 call(s)  $0.0000
    train_code_skill               4 call(s)  $0.0000
    train_prompt_skill             4 call(s)  $0.0000
    validate_prompt_skill         19 call(s)  $0.0000

  NOTE: ran in MOCK mode (no ANTHROPIC_API_KEY). Set the key and re-run
        for the real-API proof; the pipeline is identical either way.

```

## `python scripts/orchestrator_hook_demo.py --mock`

```
==============================================================================
  Orchestrator hook — does the planner see new tools after growth?
==============================================================================

LEAD planner BEFORE growth:
  actions      : ['dispatch_nav', 'direct_tool', 'capture_image', 'report_done', 'abort']
  direct tools : ['observe', 'move', 'turn', 'look_around', 'capture_image', 'check_target_depth', 'done']
  exec prompts : (none)

--- scene ['green clock', 'table'] on drone_figs ---
  planner AFTER growth — direct tools: ['observe', 'move', 'turn', 'look_around', 'capture_image', 'check_target_depth', 'done', 'target_disambiguation_among_look_alikes_classifier']
  planner AFTER growth — exec prompts: ['premature_success_on_small_or_cluttered_targets_prompt']
  NEW capabilities the planner gained:
    + direct_tool   'target_disambiguation_among_look_alikes_classifier'   <-- did not exist before growth
    + executor_prompt 'premature_success_on_small_or_cluttered_targets_prompt'  <-- specialised strategy injected
  example tool spec the planner now receives ('target_disambiguation_among_look_alikes_classifier'):
    name        : target_disambiguation_among_look_alikes_classifier
    integration : direct_tool
    input_schema: {'type': 'object', 'properties': {'clip_feature': {'type': 'array', 'items': {'type': 'number'}, 'description': 'image-region CLIP embedding'}}, 'required': ['clip_feature']}

--- scene ['chair', 'sofa'] on ground_habitat ---
  planner AFTER growth — direct tools: ['observe', 'move', 'turn', 'look_around', 'capture_image', 'check_target_depth', 'done', 'collision_recovery_in_clutter_code', 'unproductive_scanning_without_progress_code', 'target_disambiguation_among_look_alikes_classifier']
  planner AFTER growth — exec prompts: (none)
  NEW capabilities the planner gained:
    + direct_tool   'collision_recovery_in_clutter_code'   <-- did not exist before growth
    + direct_tool   'unproductive_scanning_without_progress_code'   <-- did not exist before growth
    + direct_tool   'target_disambiguation_among_look_alikes_classifier'   <-- did not exist before growth
  example tool spec the planner now receives ('collision_recovery_in_clutter_code'):
    name        : collision_recovery_in_clutter_code
    integration : direct_tool
    input_schema: {'type': 'object', 'properties': {'target_bbox': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4, 'description': 'normalised [x1,y1,x2,y2] of the target'}, 'depth_p25_m': {'type': 'number', 'description': 'P25 depth in the bbox (metres)'}}, 'required': ['target_bbox', 'depth_p25_m']}

==============================================================================
  Proof: the planner's tool surface strictly grows. Failures -> skills -> tools.
==============================================================================
```
