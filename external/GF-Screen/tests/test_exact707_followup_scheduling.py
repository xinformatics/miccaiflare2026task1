import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "run_candidate_refiner_v3_weak707_training_20260823.sh"
SOURCE = ROOT / "run_candidate_refiner_v4_source_condition_fold0_20260823.sh"
SOURCE_ROUTED = ROOT / "run_candidate_refiner_v4_source_routed_oof_20260824.sh"
SMALL = ROOT / "run_candidate_refiner_v4_small_lesion_fold0_20260823.sh"
UNCERTAIN = ROOT / "run_candidate_refiner_v4_uncertain_negative_fold0_20260824.sh"
PROMPT_POOL = ROOT / "run_candidate_refiner_v4_prompt_pool_fold0_20260824.sh"
FOCUS = ROOT / "run_focus_dualwindow_weak707_pilot_20260823.sh"
FOCUS_SELECTOR = ROOT / "select_focus_dualwindow_weak707_pilot.py"
FOCUS_MIXED_PILOT_SELECTOR = ROOT / "select_focus_b32_source_hybrid_pilot.py"
FOCUS_FIVEFOLD = ROOT / "run_focus_dualwindow_weak707_fivefold_20260823.sh"
FOCUS_FOLD4_ACCELERATOR = ROOT / "run_focus_dualwindow_fold4_accelerator_20260824.sh"
FOCUS_DISC = ROOT / "run_focus_dualwindow_discriminative_fold0_20260824.sh"
FOCUS_DISC_FIVEFOLD = ROOT / "run_focus_dualwindow_discriminative_fivefold_20260824.sh"
FOCUS_DISC_OOF = ROOT / "run_focus_dualwindow_discriminative_fullvolume_oof_20260824.sh"
FOCUS_DISC_PUBLIC = ROOT / "run_focus_dualwindow_discriminative_public_20260824.sh"
FOCUS_DISC_SAVE = ROOT / "save_focus_dualwindow_discriminative_candidate_20260824.sh"
FOCUS_FUSION = ROOT / "run_focus_classifier_fusion_oof_20260824.sh"
FOCUS_FUSION_FULLVOLUME = ROOT / "run_focus_classifier_fusion_fullvolume_oof_20260824.sh"
FOCUS_FUSION_PUBLIC = ROOT / "run_focus_classifier_fusion_public_20260824.sh"
FOCUS_FUSION_SAVE = ROOT / "save_focus_classifier_fusion_candidate_20260824.sh"
FOCUS_MIXED_OOF = ROOT / "run_focus_dualwindow_mixed_fullvolume_oof_20260824.sh"
FOCUS_MIXED_PUBLIC = ROOT / "run_focus_dualwindow_mixed_public_20260824.sh"
FOCUS_MIXED_SAVE = ROOT / "save_focus_dualwindow_mixed_candidate_20260824.sh"
FOCUS_EPOCH_ROUTE_OOF = ROOT / "run_focus_dualwindow_epoch_routed_fullvolume_oof_20260824.sh"
FOCUS_GENERIC_FULL_VOLUME = ROOT / "run_focus_candidate_refiner_fullvolume_oof_20260823.sh"
SOURCE_CLASSIFIER = ROOT / "run_source_routed_classifier_oof_20260824.sh"
TTA = ROOT / "run_focus_tta_champion_after_primary_20260824.sh"
TTA_PUBLIC = ROOT / "run_focus_tta_champion_public_20260824.sh"
TTA_SAVE = ROOT / "save_focus_tta_champion_candidate_20260824.sh"
TTA_FALLBACK_SAVE = ROOT / "save_focus_tta_evidence_floor_after_priority_20260824.sh"
TTA_PREFLIGHT = ROOT / "preflight_focus_tta_deployment_20260824.sh"
TTA_COMPARE = ROOT / "compare_focus_deployment_metrics.py"
DISCOVERY = ROOT / "run_candidate_refiner_v4_discovery_specialist_fold0_20260824.sh"
FULL_VOLUME = ROOT / "run_candidate_refiner_v4_variant_fullvolume_oof_20260824.sh"
PRIMARY_FULL_VOLUME = ROOT / "run_candidate_refiner_v3_weak707_fullvolume_oof_20260823.sh"
ENDPOINT_CONFIRMATION = ROOT / "run_candidate_refiner_endpoint_confirmation_20260824.sh"
FIVEFOLD = ROOT / "run_candidate_refiner_v4_variant_fivefold_20260824.sh"
VARIANT_CHAIN = ROOT / "run_candidate_refiner_v4_variant_public_save_20260824.sh"
PUBLIC = ROOT / "run_candidate_refiner_v3_weak707_public_20260823.sh"
SAVE = ROOT / "save_candidate_refiner_v3_weak707_candidate_20260823.sh"
UNION_NATIVE_OOF = ROOT / "run_candidate_refiner_union_native_fullvolume_oof_20260823.sh"
PRUNE_COMPOSITION = ROOT / "run_candidate_refiner_v4_prune_composition_oof_20260824.sh"
PRUNE_CHAIN = ROOT / "run_candidate_refiner_v4_prune_public_save_20260824.sh"
FOLD0_SOURCE_WATCHER = ROOT / "watch_candidate_refiner_fold0_source_audits_20260824.sh"
QUALIFICATION = ROOT / "run_candidate_refiner_v3_weak707_qualification_20260823.sh"
FAILURE_AUDIT = ROOT / "run_candidate_refiner_v4_exact707_failure_audit_20260824.sh"


class Exact707FollowupSchedulingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.source_routed = SOURCE_ROUTED.read_text(encoding="utf-8")
        cls.primary = PRIMARY.read_text(encoding="utf-8")
        cls.small = SMALL.read_text(encoding="utf-8")
        cls.uncertain = UNCERTAIN.read_text(encoding="utf-8")
        cls.prompt_pool = PROMPT_POOL.read_text(encoding="utf-8")
        cls.focus = FOCUS.read_text(encoding="utf-8")
        cls.focus_selector = FOCUS_SELECTOR.read_text(encoding="utf-8")
        cls.focus_mixed_pilot_selector = FOCUS_MIXED_PILOT_SELECTOR.read_text(
            encoding="utf-8"
        )
        cls.focus_fivefold = FOCUS_FIVEFOLD.read_text(encoding="utf-8")
        cls.focus_fold4_accelerator = FOCUS_FOLD4_ACCELERATOR.read_text(
            encoding="utf-8"
        )
        cls.focus_disc = FOCUS_DISC.read_text(encoding="utf-8")
        cls.focus_disc_fivefold = FOCUS_DISC_FIVEFOLD.read_text(encoding="utf-8")
        cls.focus_disc_oof = FOCUS_DISC_OOF.read_text(encoding="utf-8")
        cls.focus_disc_public = FOCUS_DISC_PUBLIC.read_text(encoding="utf-8")
        cls.focus_disc_save = FOCUS_DISC_SAVE.read_text(encoding="utf-8")
        cls.focus_fusion = FOCUS_FUSION.read_text(encoding="utf-8")
        cls.focus_fusion_fullvolume = FOCUS_FUSION_FULLVOLUME.read_text(encoding="utf-8")
        cls.focus_fusion_public = FOCUS_FUSION_PUBLIC.read_text(encoding="utf-8")
        cls.focus_fusion_save = FOCUS_FUSION_SAVE.read_text(encoding="utf-8")
        cls.focus_mixed_oof = FOCUS_MIXED_OOF.read_text(encoding="utf-8")
        cls.focus_mixed_public = FOCUS_MIXED_PUBLIC.read_text(encoding="utf-8")
        cls.focus_mixed_save = FOCUS_MIXED_SAVE.read_text(encoding="utf-8")
        cls.focus_epoch_route_oof = FOCUS_EPOCH_ROUTE_OOF.read_text(encoding="utf-8")
        cls.focus_generic_full_volume = FOCUS_GENERIC_FULL_VOLUME.read_text(
            encoding="utf-8"
        )
        cls.source_classifier = SOURCE_CLASSIFIER.read_text(encoding="utf-8")
        cls.tta = TTA.read_text(encoding="utf-8")
        cls.tta_public = TTA_PUBLIC.read_text(encoding="utf-8")
        cls.tta_save = TTA_SAVE.read_text(encoding="utf-8")
        cls.tta_fallback_save = TTA_FALLBACK_SAVE.read_text(encoding="utf-8")
        cls.tta_preflight = TTA_PREFLIGHT.read_text(encoding="utf-8")
        cls.tta_compare = TTA_COMPARE.read_text(encoding="utf-8")
        cls.discovery = DISCOVERY.read_text(encoding="utf-8")
        cls.full_volume = FULL_VOLUME.read_text(encoding="utf-8")
        cls.primary_full_volume = PRIMARY_FULL_VOLUME.read_text(encoding="utf-8")
        cls.endpoint_confirmation = ENDPOINT_CONFIRMATION.read_text(encoding="utf-8")
        cls.fivefold = FIVEFOLD.read_text(encoding="utf-8")
        cls.variant_chain = VARIANT_CHAIN.read_text(encoding="utf-8")
        cls.public = PUBLIC.read_text(encoding="utf-8")
        cls.save = SAVE.read_text(encoding="utf-8")
        cls.union_native_oof = UNION_NATIVE_OOF.read_text(encoding="utf-8")
        cls.prune_composition = PRUNE_COMPOSITION.read_text(encoding="utf-8")
        cls.prune_chain = PRUNE_CHAIN.read_text(encoding="utf-8")
        cls.fold0_source_watcher = FOLD0_SOURCE_WATCHER.read_text(encoding="utf-8")
        cls.qualification = QUALIFICATION.read_text(encoding="utf-8")
        cls.failure_audit = FAILURE_AUDIT.read_text(encoding="utf-8")

    def test_recommended_b32_ablations_wait_for_primary_save_terminal(self):
        marker = "SAVE_ONLY_COMPLETE.txt"
        self.assertIn(marker, self.source)
        self.assertIn(marker, self.small)
        self.assertIn(marker, self.uncertain)

    def test_short_public_tta_precedes_long_fold0_ablations(self):
        marker = "TTA_SAVE_TERMINAL.json"
        self.assertNotIn("SAVE_ONLY_COMPLETE.txt", self.tta)
        self.assertIn("waiting for TTA deployment preflight", self.tta)
        self.assertIn(marker, self.small)
        self.assertIn(marker, self.discovery)
        self.assertIn('"hidden_read": False', self.tta)
        self.assertIn('"submission_created": False', self.tta)
        self.assertIn('"uploaded": False', self.tta)
        self.assertNotIn("codabench", self.tta.lower())

    def test_exploratory_tta_fallback_is_save_only_and_lower_priority(self):
        script = self.tta_fallback_save
        marker = "TTA_FALLBACK_SAVE_TERMINAL.json"
        self.assertIn(marker, self.small)
        self.assertIn(marker, self.discovery)
        self.assertIn("POST_HOC_EVIDENCE_FLOOR_NOT_PRIMARY_PROMOTION", script)
        self.assertIn("higher_priority_inference_markers", script)
        self.assertIn("TRAINING_COMPLETE.json", script)
        self.assertIn("fold4/work_bbox24/INFERENCE_COMPLETE.json", script)
        self.assertIn(marker, self.focus)
        self.assertIn("--focus_tta_axes xy", script)
        self.assertIn("--expected-count 100", script)
        self.assertIn('"automatic_upload_authorized":False', script)
        self.assertIn('"uploaded":False', script)
        self.assertNotIn("codabench.org", script.lower())

    def test_tta_scores_the_identical_final_champion_deployment(self):
        script = self.tta_public
        self.assertIn("biomedparse_fusion.py", script)
        self.assertIn("screen_disconnected_guided_additions.py", script)
        self.assertIn("--max-mm3 2000", script)
        self.assertIn("--min-support-fraction 0.25", script)
        self.assertIn("--min-support-mm3 100", script)
        self.assertIn("new2k_s25", script)
        self.assertIn("gf_champ_new2k_s25", script)
        self.assertNotIn("Validation-Hidden", script)
        self.assertNotIn("package_codabench", script)
        self.assertNotIn("codabench", script.lower())

    def test_tta_hidden_materialization_is_strictly_public_gated_and_save_only(self):
        script = self.tta_save
        self.assertIn("PUBLIC_TTA_TERMINAL.json", script)
        self.assertIn("PROMOTE_TTA_HIDDEN", script)
        self.assertIn("select_focus_tta_candidate.py", script)
        self.assertIn("--focus_tta_axes", script)
        self.assertIn("biomedparse_fusion.py", script)
        self.assertIn("screen_disconnected_guided_additions.py", script)
        self.assertIn("--expected-count 100", script)
        self.assertIn('"uploaded":False', script)
        self.assertNotIn("codabench.org", script.lower())

    def test_tta_selector_failure_closes_as_keep_champion_terminal(self):
        script = self.tta_save
        self.assertIn('selection_status=0', script)
        self.assertIn('selector rejected an invalid or non-equivalent baseline', script)
        self.assertIn('"decision":"KEEP_CHAMPION"', script)
        self.assertIn('"hidden_read":False', script)

    def test_tta_preflight_proves_bounded_champion_deployment_without_hidden_data(self):
        script = self.tta_preflight
        self.assertIn("compare_focus_deployment_metrics.py", script)
        self.assertIn("exact_metric_rows", self.tta_compare)
        self.assertIn("deployment_equivalent", self.tta_compare)
        self.assertIn("MAX_TOTAL_ABS_VOXEL_DELTA", self.tta_compare)
        self.assertIn("biomedparse_fusion.py", script)
        self.assertIn("screen_disconnected_guided_additions.py", script)
        self.assertIn("PREFLIGHT_COMPLETE.json", script)
        self.assertNotIn("Validation-Hidden", script)
        self.assertNotIn("package_codabench", script)

    def test_source_conditioning_is_authorized_by_final_source_audit(self):
        self.assertIn("SOURCE_DOMAIN_AUDIT.json", self.source)
        self.assertIn("test_source_conditioning", self.source)
        self.assertIn("source audit/index identity mismatch", self.source)
        self.assertIn(
            "candidate_refiner_v4_cell707_discovery_specialist_saved_20260824/VARIANT_CHAIN_TERMINAL.json",
            self.source,
        )

    def test_targeted_ablations_use_distinct_gpus_and_wait_for_availability(self):
        self.assertIn('[[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]]', self.source)
        self.assertIn('gpu="${SMALL_LESION_GPU:-1}"', self.small)
        self.assertIn('[[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]]', self.small)
        self.assertIn('[[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]]', self.uncertain)

    def test_prompt_pool_is_source_gated_and_precedes_small_lesion_ablation(self):
        self.assertIn("--prompt-masked-classifier", self.prompt_pool)
        self.assertIn("audit_candidate_refiner_fold_sources.py", self.prompt_pool)
        self.assertIn("select_candidate_refiner_prompt_pool.py", self.prompt_pool)
        self.assertIn("baseline_best_classification_source.json", self.prompt_pool)
        self.assertIn("candidate_best_classification_source.json", self.prompt_pool)
        self.assertIn('plan="${decision_dir}/PLAN_V3.json"', self.prompt_pool)
        self.assertIn('superseded_plan="${decision_dir}/PLAN_V2.json"', self.prompt_pool)
        self.assertIn('"prompt_channel":2', self.prompt_pool)
        self.assertIn('"prompt_channel_semantic":"seed_support"', self.prompt_pool)
        self.assertIn("adaptive_average_occupancy", self.prompt_pool)
        self.assertIn("superseded_plan_sha256", self.prompt_pool)
        self.assertIn("frozen prompt-pool plan does not match", self.prompt_pool)
        self.assertIn('"plan_sha256":sha(plan)', self.prompt_pool)
        self.assertIn("prompt_pool_terminal", self.small)
        self.assertNotIn("Validation-Public", self.prompt_pool)
        self.assertNotIn("Validation-Hidden", self.prompt_pool)
        self.assertNotIn("package_codabench", self.prompt_pool)
        for script in (self.fivefold, self.full_volume, self.variant_chain):
            self.assertIn("prompt_pool", script)

    def test_primary_oof_reuses_validated_inference_batch_capacity(self):
        # The same B32 checkpoint is exhaustively validated at batch 32 during
        # training. Keep post-training OOF at that proven inference-only size
        # instead of the historical batch 8 bottleneck.
        self.assertIn('--batch-size 32 --workers 2', self.primary_full_volume)
        self.assertIn('--uncompressed-predictions', self.primary_full_volume)
        self.assertIn('--uncompressed-predictions --resume', self.primary_full_volume)
        self.assertIn('--require-inference-plan', self.primary_full_volume)
        self.assertIn('--workers 16', self.primary_full_volume)
        self.assertIn(
            'payload.get("prediction_cache_compressed") is False',
            self.primary_full_volume,
        )

    def test_primary_public_and_hidden_inference_are_plan_bound_and_resumable(self):
        for script in (self.public, self.save):
            self.assertIn('--workers 2 --resume', script)
            self.assertIn('--require-inference-plan', script)

    def test_endpoint_confirmation_is_conditional_exact707_and_label_internal(self):
        script = self.endpoint_confirmation
        self.assertIn("ENDPOINT_CONFIRMATION_PLAN.json", script)
        self.assertIn("FULL_VOLUME_OOF_COMPLETE.json", script)
        self.assertIn("RUN_ENDPOINT_CONFIRMATION", script)
        for threshold in ("0.95", "0.96", "0.97", "0.98"):
            self.assertIn(threshold, script)
        self.assertIn("--bootstrap 20000", script)
        self.assertIn("staging/all/labels", script)
        self.assertNotIn("Validation-Public", script)
        self.assertNotIn("Validation-Hidden", script)
        self.assertNotIn("package_codabench", script)
        self.assertNotIn("codabench", script.lower())

    def test_speculative_uncertain_negative_waits_for_targeted_variant_chains(self):
        self.assertIn("source_condition_terminal", self.uncertain)
        self.assertIn("smallloss_terminal", self.uncertain)
        self.assertIn("higher-priority ablation terminals", self.uncertain)
        self.assertIn("precedence_terminal", self.variant_chain)
        self.assertIn("source_conditioning_saved_20260824/VARIANT_CHAIN_TERMINAL.json", self.variant_chain)
        self.assertIn("small_loss_saved_20260824/VARIANT_CHAIN_TERMINAL.json", self.variant_chain)

    def test_focus_is_prioritized_directly_after_primary_and_tta_gates(self):
        self.assertIn("primary_pipeline_complete", self.focus)
        self.assertIn("tta_terminal", self.focus)
        self.assertIn("FOCUS_DUALWINDOW_ALLOW_EARLY_GPU", self.focus)
        self.assertIn('if [[ "${allow_early}" == 1 ]]', self.focus)
        self.assertIn('"allow_early_gpu":allow_early == "1"', self.focus)
        self.assertIn('"physical_gpu":int(gpu)', self.focus)
        self.assertNotIn("smallloss_terminal", self.focus)
        self.assertNotIn("source_condition_terminal", self.focus)
        self.assertNotIn("uncertain_negative_terminal", self.focus)
        self.assertLess(self.focus.index("tta_terminal"), self.focus.index("FOCUS_DUALWINDOW_GPU"))

    def test_focus_fullvolume_serializes_large_cache_allocation_after_model_gate(self):
        script = self.focus_generic_full_volume
        self.assertIn("FOCUS_OOF_STORAGE_LOCK", script)
        self.assertIn("FOCUS_OOF_MINIMUM_FREE_GIB:-180", script)
        self.assertIn('exec 8>"${storage_lock}"', script)
        self.assertIn("flock 8", script)
        self.assertIn('df -Pk "${output}"', script)
        self.assertIn("required_free_kib", script)
        self.assertLess(
            script.index('if [[ "${focus_status}" != COMPLETE'),
            script.index("flock 8"),
        )
        self.assertLess(script.index("flock 8"), script.index("apply_candidate_refiner.py"))

    def test_small_loss_waits_for_complete_higher_capacity_focus_chain(self):
        self.assertIn("focus_terminal", self.small)
        self.assertIn(
            "focus_dualwindow_cell707_saved_20260823/SAVE_ONLY_COMPLETE.txt",
            self.small,
        )
        self.assertLess(self.small.index("focus_terminal"), self.small.index("SMALL_LESION_GPU"))

    def test_focus_fivefold_preserves_hash_bound_early_source_qualification(self):
        script = self.focus_fivefold
        self.assertIn("MIXED_EARLY_DECISION.json", script)
        self.assertIn("checkpoint_epoch_0001.pt", script)
        self.assertIn("validation_predictions_epoch_0001.npz", script)
        self.assertIn("early checkpoint hash mismatch", script)
        self.assertIn("early prediction hash mismatch", script)
        self.assertIn('set(payload.get("eligible_focus_groups",[]))', script)
        self.assertIn('payload.get("fixed_b32_groups")!=["mature"]', script)
        self.assertIn("mixed_early_pilot_decision", script)

    def test_focus_fold4_accelerator_is_recipe_identical_guarded_and_local_only(self):
        script = self.focus_fold4_accelerator
        for argument in (
            "--folds 5 --fold 4 --epochs 10 --batch-size 8 --workers 2",
            "--lr 3e-5 --weight-decay 1e-5 --loss-recipe release_dice_ce",
            "--sampling-temperature 0.5 --samples-per-epoch 2048",
            "--freeze-encoder --input-mode soft_lung_seed --seed 20260823",
        ):
            self.assertIn(argument, script)
        self.assertIn("PROMOTE_TO_FIVEFOLD", script)
        self.assertIn("PROMOTE_MIXED_TO_FIVEFOLD", script)
        self.assertIn('[[ ! -s "${output}/COMPLETE.txt"', script)
        self.assertIn('[[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]]', script)
        self.assertIn('"public_read":False,"hidden_read":False', script)
        self.assertNotIn("Validation-Public", script)
        self.assertNotIn("Validation-Hidden", script)
        self.assertNotIn("package_codabench", script)
        self.assertNotIn("codabench", script.lower())

    def test_focus_plan_and_selector_include_same_threshold_ceiling(self):
        self.assertIn('"segmentation_threshold_max":0.95', self.focus.replace(" ", ""))
        self.assertIn(
            '"segmentation_threshold_max":0.95',
            self.focus_selector.replace(" ", ""),
        )

    def test_source_specific_focus_escape_gate_is_predeclared_and_held_out(self):
        compact = self.focus.replace(" ", "")
        selector = self.focus_mixed_pilot_selector.replace(" ", "")
        self.assertIn('"minimum_case_dice_gain_over_b32":0.02', compact)
        self.assertIn('"minimum_cohort_dice_gain_over_b32":0.01', compact)
        self.assertIn('"maximum_focus_negative_nonempty_rate":0.95', compact)
        self.assertIn('"mature_route":"b32"', compact)
        self.assertIn('"minimum_case_dice_gain_over_b32":0.02', selector)
        self.assertIn('"minimum_cohort_dice_gain_over_b32":0.01', selector)
        self.assertIn("grouped_fold_rows(rows,5,0,20260823)", selector)
        self.assertIn("negative_nonempty_grid", selector)
        self.assertNotIn("Validation-Public", self.focus_mixed_pilot_selector)
        self.assertNotIn("Validation-Hidden", self.focus_mixed_pilot_selector)

    def test_global_or_source_specific_focus_can_reach_fivefold(self):
        self.assertIn("PROMOTE_TO_FIVEFOLD", self.focus_fivefold)
        self.assertIn("PROMOTE_MIXED_TO_FIVEFOLD", self.focus_fivefold)
        self.assertIn("MIXED_DECISION.json", self.focus_fivefold)
        self.assertIn("decoupled_oof", self.focus_fivefold)
        self.assertIn("source-pooled B32 contours differ from the primary", self.focus_fivefold)

    def test_discriminative_focus_runs_only_when_both_frozen_routes_fail(self):
        self.assertIn("frozen_mixed_decision", self.focus_disc)
        self.assertIn("PROMOTE_MIXED_TO_FIVEFOLD", self.focus_disc)

    def test_discriminative_focus_is_a_local_only_rejected_pilot_fallback(self):
        script = self.focus_disc
        self.assertIn('frozen_status}" != KEEP_B32', script)
        self.assertIn("--encoder-lr-scale 0.1", script)
        self.assertNotIn("--freeze-encoder", script)
        self.assertIn('"public_read":False,"hidden_read":False', script)
        self.assertNotIn("Validation-Hidden", script)
        self.assertNotIn("package_codabench", script)
        self.assertNotIn("codabench", script.lower())
        self.assertNotIn("package_codabench", script)
        self.assertNotIn("codabench", script.lower())

    def test_discriminative_focus_has_a_strict_full_deployment_chain(self):
        self.assertIn("PROMOTE_TO_FIVEFOLD", self.focus_disc_fivefold)
        self.assertIn("--encoder-lr-scale 0.1", self.focus_disc_fivefold)
        self.assertIn("FOCUS_TRAIN_TERMINAL", self.focus_disc_oof)
        self.assertIn("FOCUS_OOF_EXPECTED_CASES=707", self.focus_disc_oof)
        self.assertIn("WEAK707_PUBLIC_REQUIRED_DECISION=PROMOTE", self.focus_disc_public)
        self.assertIn("WEAK707_PUBLIC_PROPOSAL_CACHE_ROOT", self.focus_disc_public)
        self.assertIn("WEAK707_SAVE_ALLOW_ANCHOR_FALLBACK=0", self.focus_disc_save)
        self.assertIn("WEAK707_SAVE_PROPOSAL_CACHE_ROOT", self.focus_disc_save)
        self.assertNotIn("codabench", self.focus_disc_oof.lower())
        self.assertNotIn("codabench", self.focus_disc_public.lower())
        self.assertNotIn("codabench", self.focus_disc_save.lower())

    def test_focus_classifier_fusion_is_cross_fitted_and_local_only(self):
        script = self.focus_fusion
        self.assertIn("analyze_focus_classifier_fusion_oof.py", script)
        self.assertIn("validation_predictions_best_classification.npz", script)
        self.assertIn("validation_predictions_best.npz", script)
        self.assertIn("--bootstrap 2000", script)
        self.assertIn('"public_read":False,"hidden_read":False', script)
        self.assertNotIn("Validation-Hidden", script)
        self.assertNotIn("package_codabench", script)
        self.assertNotIn("codabench", script.lower())

    def test_qualified_focus_classifier_fusion_has_strict_deployment_chain(self):
        self.assertIn("FOCUS_OOF_CLASSIFIER_FUSION", self.focus_fusion_fullvolume)
        self.assertIn("FOCUS_OOF_EXPECTED_CASES=707", self.focus_fusion_fullvolume)
        self.assertIn(
            "FOCUS_OOF_CLASSIFIER_CHECKPOINT_NAME=checkpoint_best_classification.pt",
            self.focus_fusion_fullvolume,
        )
        self.assertIn("WEAK707_PUBLIC_CLASSIFIER_FUSION", self.focus_fusion_public)
        self.assertIn(
            "WEAK707_PUBLIC_CLASSIFIER_CHECKPOINT_NAME=checkpoint_best_classification.pt",
            self.focus_fusion_public,
        )
        self.assertNotIn("Validation-Hidden", self.focus_fusion_public)
        self.assertNotIn("package_codabench", self.focus_fusion_public)
        self.assertIn("WEAK707_SAVE_CLASSIFIER_FUSION", self.focus_fusion_save)
        self.assertIn(
            "WEAK707_SAVE_CLASSIFIER_CHECKPOINT_NAME=checkpoint_best_classification.pt",
            self.focus_fusion_save,
        )
        self.assertIn("WEAK707_SAVE_ALLOW_ANCHOR_FALLBACK=0", self.focus_fusion_save)
        self.assertNotIn("codabench.org", self.focus_fusion_save.lower())

    def test_mixed_capacity_focus_route_is_exact707_local_only(self):
        self.assertIn("FOCUS_OOF_MIXED_ROUTE_DECISION", self.focus_mixed_oof)
        self.assertIn("FOCUS_OOF_EXPECTED_CASES=707", self.focus_mixed_oof)
        self.assertIn("focus_b32_source_hybrid_decision.json", self.focus_mixed_oof)
        self.assertNotIn("Validation-Public", self.focus_mixed_oof)
        self.assertNotIn("Validation-Hidden", self.focus_mixed_oof)
        self.assertNotIn("package_codabench", self.focus_mixed_oof)

    def test_mixed_capacity_route_has_strict_public_and_save_consumers(self):
        self.assertIn("WEAK707_PUBLIC_REQUIRED_DECISION=PROMOTE", self.focus_mixed_public)
        self.assertIn("WEAK707_PUBLIC_MIXED_ROUTE_DECISION", self.focus_mixed_public)
        self.assertNotIn("Validation-Hidden", self.focus_mixed_public)
        self.assertNotIn("package_codabench", self.focus_mixed_public)
        self.assertIn("WEAK707_SAVE_MIXED_ROUTE_DECISION", self.focus_mixed_save)
        self.assertIn("WEAK707_SAVE_ALLOW_ANCHOR_FALLBACK=0", self.focus_mixed_save)
        self.assertNotIn("codabench.org", self.focus_mixed_save.lower())
        for script in (self.public, self.save):
            self.assertIn("focus_b32_hybrid_route_selection.py", script)
            self.assertIn('--candidate-contour-route "b32=', script)
            self.assertIn('--focus-contour-route "focus=', script)
            self.assertIn("focus_routed_input_mode", script)

    def test_focus_epoch_routing_waits_behind_mixed_capacity_chain(self):
        self.assertIn(
            "focus_dualwindow_mixed_saved_20260824/SAVE_ONLY_COMPLETE.txt",
            self.focus_epoch_route_oof,
        )
        self.assertIn("higher-priority mixed B32/Focus chain terminal", self.focus_epoch_route_oof)

    def test_source_classifier_routing_is_patch_oof_gated_before_deployment(self):
        self.assertIn("analyze_source_routed_classifier_oof.py", self.source_classifier)
        self.assertIn("build_source_routed_classifier_patch_oof.py", self.source_classifier)
        self.assertIn("TRAINING_COMPLETE.json", self.source_classifier)
        self.assertIn("LIVE_SOURCE_AUDITS_COMPLETE.json", self.source_classifier)
        self.assertIn("all 55 exact707 scored-epoch source audits", self.source_classifier)
        self.assertIn('decoupled_terminal="${decoupled}/TERMINAL.json"', self.source_classifier)
        self.assertIn("QUALIFY_CLASSIFIER_ROUTED_FULL_VOLUME", self.source_classifier)
        self.assertIn("--bootstrap 2000", self.source_classifier)
        self.assertNotIn("Validation-Public", self.source_classifier)
        self.assertNotIn("Validation-Hidden", self.source_classifier)
        self.assertNotIn("package_codabench", self.source_classifier)

    def test_qualified_source_classifier_route_has_strict_priority_lane(self):
        self.assertIn("classifier_routed)", self.full_volume)
        self.assertIn("QUALIFY_CLASSIFIER_ROUTED_FULL_VOLUME", self.full_volume)
        self.assertIn("candidate_refiner_classifier_route_selection.py", self.full_volume)
        self.assertIn("waiting for primary save and direct decoupled chain", self.full_volume)
        self.assertIn('--batch-size 32 --workers 2 --resume', self.full_volume)
        self.assertIn("classifier_routed)", self.variant_chain)
        self.assertIn("WEAK707_PUBLIC_CLASSIFIER_ROUTE_DECISION", self.variant_chain)
        self.assertIn("WEAK707_SAVE_CLASSIFIER_ROUTE_DECISION", self.variant_chain)
        for script in (self.public, self.save):
            self.assertIn("candidate_refiner_classifier_route_selection.py", script)
            self.assertIn("--classifier-route", script)
            self.assertIn("--source-classifier-route", script)

    def test_focus_preflight_is_hash_bound_through_selection(self):
        self.assertIn('preflight="${analysis}/PREFLIGHT.json"', self.focus)
        self.assertIn('"preflight_sha256":sha(preflight)', self.focus)
        self.assertIn("Focus checkpoint preflight changed after planning", self.focus_selector)
        self.assertIn("Focus preflight/checkpoint identity mismatch", self.focus_selector)

    def test_variant_full_volume_reuses_identical_prepared_cases_but_reinfers(self):
        self.assertIn('cp "${source}/manifest.json"', self.full_volume)
        self.assertIn("--allow-external-patches --require-predictions", self.full_volume)
        self.assertIn('apply_candidate_refiner.py" infer', self.full_volume)
        self.assertIn("select_segmentation_threshold_sweep.py", self.full_volume)

    def test_primary_decoupled_pair_has_a_direct_matched_deployment_lane(self):
        self.assertIn("primary_decoupled)", self.full_volume)
        self.assertIn("QUALIFY_DECOUPLED_FULL_VOLUME", self.full_volume)
        self.assertIn('"selected_strategy":"decoupled"', self.full_volume)
        self.assertIn(
            '"contour_checkpoint_filename":"checkpoint_best_segmentation.pt"',
            self.full_volume,
        )
        self.assertIn('--contour-checkpoint "${contour_checkpoint}"', self.full_volume)
        self.assertIn("primary_decoupled)", self.variant_chain)
        self.assertIn(
            '[[ "${variant}" == primary_decoupled ]] && deploy_selection="${oof}/checkpoint_selection.json"',
            self.variant_chain,
        )

    def test_decoupled_selection_reaches_public_and_hidden_inference(self):
        for script in (self.public, self.save):
            self.assertIn('get("contour_checkpoint_filename")', script)
            self.assertIn('isinstance(v,str) and v else "NONE"', script)
            self.assertIn('--contour-checkpoint "${contour}"', script)
            self.assertIn("selection and external contour summary are mutually exclusive", script)

    def test_decoupled_deployment_reuses_label_blind_proposal_caches(self):
        self.assertIn("WEAK707_PUBLIC_PROPOSAL_CACHE_ROOT", self.public)
        self.assertIn("WEAK707_SAVE_PROPOSAL_CACHE_ROOT", self.save)
        self.assertIn(
            'public_proposal_cache="${root}/public_sweeps/candidate_refiner_v4_cell707_20260823"',
            self.variant_chain,
        )
        self.assertIn(
            'hidden_proposal_cache="${root}/hidden_predictions/candidate_refiner_v4_cell707_20260823"',
            self.variant_chain,
        )
        self.assertIn('export WEAK707_PUBLIC_PROPOSAL_CACHE_ROOT=', self.variant_chain)
        self.assertIn('export WEAK707_SAVE_PROPOSAL_CACHE_ROOT=', self.variant_chain)

    def test_direct_decoupled_lane_precedes_discovery_training(self):
        marker = (
            "candidate_refiner_v4_cell707_decoupled_saved_20260824/"
            "VARIANT_CHAIN_TERMINAL.json"
        )
        self.assertIn(marker, self.discovery)
        self.assertIn("decoupled, and source-routed chain terminals", self.discovery)
        self.assertIn("waiting for primary exact-707 save-or-skip terminal", self.full_volume)

    def test_primary_full_volume_waits_briefly_for_dual_gpu_dispatch(self):
        script = self.primary_full_volume
        self.assertIn("CELL707_OOF_MAX_IDLE_GPU_MEMORY_MIB:-2048", script)
        self.assertIn('"${max_idle_gpu_memory_mib}" -le 4096', script)
        self.assertIn("second_gpu_deadline=$((SECONDS + 300))", script)
        self.assertIn("dual-GPU OOF dispatch", script)
        self.assertIn("infer_sequence 0 0 2 4", script)
        self.assertIn("infer_sequence 1 1 3", script)

    def test_variant_full_volume_requires_paired_dsc_and_nsd_promotion(self):
        self.assertIn("--min-dsc-gain 0.005 --min-nsd-gain 0.005", self.full_volume)
        self.assertIn("--min-win-rate 0.52", self.full_volume)
        self.assertIn("--max-catastrophic-rate 0.10", self.full_volume)
        self.assertIn('"QUALIFY_FOR_FIXED_PUBLIC" if decision=="PROMOTE"', self.full_volume)

    def test_primary_full_volume_calibrates_actual_anchor_composition(self):
        script = self.primary_full_volume
        self.assertIn('composition_root="${output}/anchor_currentgf"', script)
        self.assertIn('base="${staging}/fold${fold}/proposals/currentgf"', script)
        self.assertIn('anchor_currentgf "${composition_root}"', script)
        self.assertIn("composition_segmentation=$(tail -n 1", script)
        self.assertIn('"${composition_root}/SEGMENTATION_THRESHOLD_SELECTION.json"', script)
        self.assertIn('write_materialization_grid "${composition_segmentation}"', script)
        self.assertIn("--min-dsc-gain 0.005 --min-nsd-gain 0.005", script)
        self.assertIn('payload["calibration_deployment"]', script)
        self.assertIn('robust.get("decision")=="PROMOTE"', script)
        # Host pressure measurements bound each deployment sweep to three fold
        # workers. Keep the standalone and anchor sweeps sequential so peak
        # concurrency is three, while retaining parallel scoring afterward.
        self.assertNotIn("seg_sweep_pids=()", script)
        self.assertIn('for fold_group in "0 1 2" "3 4"', script)
        standalone = script.index("run_segmentation_sweep standalone")
        composition = script.index("run_segmentation_sweep anchor_currentgf")
        self.assertLess(standalone, composition)
        self.assertIn("score_pids=()", script)
        self.assertIn('for pid in "${score_pids[@]}"', script)

    def test_future_single_point_full_volume_grids_are_memory_bounded(self):
        for script in (self.focus_generic_full_volume, self.union_native_oof):
            self.assertIn('("0 1 2" "3 4")', script)
            self.assertNotIn('("0 1 2 3 4")', script)

    def test_downstream_oof_replays_primary_selected_deployment(self):
        for script in (self.full_volume, self.focus_generic_full_volume):
            self.assertIn('get("calibration_deployment","standalone")', script)
            self.assertIn('{"standalone","anchor_currentgf"}', script)
            self.assertIn('proposals/currentgf', script)
            self.assertIn('payload["calibration_deployment"]=deployment', script)
            self.assertIn('payload["fixed_deployment_source"]', script)

    def test_public_and_hidden_cannot_reverse_oof_selected_deployment(self):
        self.assertIn('get("calibration_deployment")', self.public)
        self.assertIn('deployments=(standalone)', self.public)
        self.assertIn('deployments=(anchor)', self.public)
        self.assertNotIn('deployments=(anchor standalone)', self.public)
        self.assertIn('expected_deployment=$("${python_bin}"', self.save)
        self.assertIn('public selection changed frozen deployment', self.save)
        self.assertIn('"calibration_deployment":"anchor_currentgf"', self.prune_composition)

    def test_scale_qualification_is_standalone_matched_and_not_deadlocked(self):
        self.assertIn("prepare_scaled_oof_comparison.py", self.qualification)
        self.assertNotIn("source_separate/TRANSFER_OOF_COMPLETE", self.qualification)
        self.assertNotIn('--baseline "source_separate=', self.qualification)
        self.assertIn('"scaled_selected_deployment"', self.qualification)
        self.assertIn('"scaled_comparison_deployment"', self.qualification)
        self.assertIn('paired_exact707_anchor_composition', self.qualification)
        self.assertIn('matched_shared210_standalone_scale', self.qualification)
        self.assertIn('all(value is True for value in checks.values())', self.qualification)

    def test_source_contour_calibration_is_symmetric_through_deployment(self):
        self.assertIn("source_segmentation_threshold_json", self.full_volume)
        self.assertIn("source_segmentation_thresholds", self.full_volume)
        self.assertIn("source-segmentation-threshold", self.public)
        self.assertIn("source-segmentation-threshold", self.save)

    def test_source_routing_selects_weak_domains_and_lung_independently(self):
        script = self.source_routed
        self.assertIn("autopet_weak autopet", script)
        self.assertIn("lndb_weak lndb", script)
        self.assertIn("lung_cells lung", script)
        self.assertIn("autopet_weak=autopet", script)
        self.assertIn("lndb_weak=lndb", script)
        self.assertIn('lung_cell${cell}=lung', script)
        self.assertIn("select_candidate_refiner_partial_source_routes.py", script)
        self.assertIn("--minimum-case-dice-gain 0.002", script)
        self.assertIn("--minimum-cohort-dice-gain 0.001", script)
        self.assertIn("validation_predictions_best_segmentation.npz", script)
        self.assertIn("PARTIAL_ROUTE_SELECTION.json", self.full_volume)
        self.assertIn("QUALIFY_GROUP_SELECTED_SOURCE_ROUTE", self.full_volume)
        for checkpoint in (
            "checkpoint_best_autopet_segmentation.pt",
            "checkpoint_best_lndb_segmentation.pt",
            "checkpoint_best_lung_segmentation.pt",
        ):
            self.assertIn(checkpoint, self.full_volume)
        self.assertIn("candidate_refiner_route_selection.py", self.full_volume)
        self.assertIn(
            "candidate_refiner_v4_cell707_saved_20260823/SAVE_ONLY_COMPLETE.txt",
            self.full_volume,
        )
        self.assertIn(
            "candidate_refiner_v4_cell707_decoupled_saved_20260824/VARIANT_CHAIN_TERMINAL.json",
            self.full_volume,
        )
        self.assertIn(
            "candidate_refiner_v4_cell707_source_routed_saved_20260824/VARIANT_CHAIN_TERMINAL.json",
            self.discovery,
        )
        self.assertLess(
            self.full_volume.index("primary_save_complete"),
            self.full_volume.index(
                "waiting for primary save and direct decoupled chain"
            ),
        )

    def test_source_deployment_modes_are_oof_selected_and_deployed_symmetrically(self):
        self.assertIn("source_deployment_modes", self.full_volume)
        self.assertIn('"epoch105":"seed","currentgf":"seed"', self.full_volume)
        self.assertIn('"epoch105":"seed_crop","currentgf":"seed_crop"', self.full_volume)
        self.assertIn("source-mode", self.public)
        self.assertIn("source-mode", self.save)

    def test_variant_public_chain_uses_variant_checkpoints_and_calibration(self):
        self.assertIn('WEAK707_PUBLIC_TRAIN_ROOT="${fivefold}"', self.variant_chain)
        self.assertIn('WEAK707_PUBLIC_RUN_PREFIX="${run_prefix}"', self.variant_chain)
        self.assertIn('WEAK707_PUBLIC_CALIBRATION="${oof}/deployment_calibration.json"', self.variant_chain)
        self.assertIn("WEAK707_PUBLIC_TRAIN_ROOT", self.public)
        self.assertIn("WEAK707_PUBLIC_RUN_PREFIX", self.public)

    def test_variant_hidden_inference_requires_strict_public_promotion(self):
        strict_select = '--summary "${public}/summary.tsv" --expected-cases 50)'
        self.assertIn(strict_select, self.variant_chain)
        self.assertLess(self.variant_chain.index(strict_select), self.variant_chain.index("WEAK707_SAVE_PUBLIC_ROOT"))
        self.assertIn("WEAK707_SAVE_ALLOW_ANCHOR_FALLBACK=0", self.variant_chain)
        self.assertIn("WEAK707_SAVE_ALLOW_ANCHOR_FALLBACK", self.save)

    def test_variant_chain_validates_but_never_uploads_exact_100_zip(self):
        self.assertIn("len(members)!=100", self.variant_chain)
        self.assertIn("archive.testzip()", self.variant_chain)
        self.assertIn('"uploaded":False', self.variant_chain)
        self.assertNotIn("codabench", self.variant_chain.lower())

    def test_saved_queue_discovers_every_controlled_ablation_manifest(self):
        self.assertIn('--manifest-root "${root}/public_sweeps"', self.save)
        self.assertIn("refresh_all_saved_candidates_20260822.py", self.save)

    def test_geometry_fallback_mode_and_gates_propagate_to_public_and_hidden(self):
        for tag in ("geomlo", "geommid", "geomhi"):
            self.assertIn(tag, self.primary_full_volume)
        self.assertIn("geometry_gates", self.primary_full_volume)
        self.assertIn('"geometry_gates":calibration.get("geometry_gates",{})', self.primary_full_volume)
        for script in (self.full_volume, self.focus_generic_full_volume):
            self.assertIn("primary_calibration", script)
            self.assertIn("geometry_gate_json", script)
            self.assertIn('--min-seed-coverage "${min_seed_coverage}"', script)
            self.assertIn("geometry gates fixed from primary exact OOF", script)
        for script in (self.public, self.save):
            self.assertIn("refined_or_seed", script)
            self.assertIn('get("geometry_gates",{})', script)
            self.assertIn('--min-seed-coverage "${min_seed_coverage}"', script)
            self.assertIn('--max-volume-ratio "${max_volume_ratio}"', script)

    def test_uncertain_negative_ablation_is_bounded_and_provenance_matched(self):
        self.assertIn("--classification-negative-weight 0.25", self.uncertain)
        self.assertIn("--negative-segmentation-weight 0.025", self.uncertain)
        self.assertIn("--negative-segmentation-weight 0.025", self.fivefold)
        self.assertIn("candidate_refiner_v4_cell707_b32_t05_fold0", self.uncertain)
        self.assertNotIn("public_sweeps/", self.uncertain.split("primary_pipeline_complete=")[0])
        for script in (self.fivefold, self.full_volume, self.variant_chain):
            self.assertIn("uncertain_negative", script)

    def test_small_lesion_cap_matches_the_authorizing_source_audit(self):
        scripts = (self.small, self.fivefold, self.full_volume, self.variant_chain)
        for script in scripts:
            self.assertIn("smallloss30", script)
            self.assertNotIn("smallloss100", script)
        self.assertIn("--small-lesion-positive-weight-cap 30", self.small)
        self.assertIn("--small-lesion-positive-weight-cap 30", self.fivefold)

    def test_exact_families_share_sparse_validation_and_dynamic_dispatch(self):
        for script in (
            self.primary,
            self.source,
            self.prompt_pool,
            self.small,
            self.uncertain,
            self.fivefold,
        ):
            self.assertIn("--validation-interval 3", script)
        self.assertIn("train_remaining_dynamic 0 1 2 3 4", self.primary)
        self.assertNotIn("train_sequence 0 0 2 4", self.primary)

    def test_fold0_source_terminal_requires_final_scored_epoch(self):
        script = self.fold0_source_watcher
        self.assertIn('checkpoint_epoch_0030.pt', script)
        self.assertIn('validation_predictions_epoch_0030.npz', script)
        self.assertIn('missing required source audit', script)
        self.assertLess(
            script.index('checkpoint_epoch_0030.pt'),
            script.index('current_selection='),
        )

    def test_exact707_failure_audit_is_post_save_internal_only(self):
        script = self.failure_audit
        self.assertIn("FULL_VOLUME_OOF_COMPLETE.json", script)
        self.assertIn("SAVE_ONLY_COMPLETE.txt", script)
        self.assertIn("analyze_lesion_errors.py", script)
        self.assertIn("summarize_candidate_refiner_failure_modes.py", script)
        self.assertIn("exact-707 audit coverage mismatch", script)
        self.assertNotIn("Validation-Public", script)
        self.assertNotIn("Validation-Hidden", script)
        self.assertNotIn("package_codabench", script)

    def test_cohort_balanced_checkpoint_is_available_to_fivefold_selection(self):
        checkpoint = "checkpoint_best_cohort_segmentation.pt"
        prediction = "validation_predictions_best_cohort_segmentation.npz"
        for script in (self.primary, self.fivefold):
            self.assertIn(checkpoint, script)
            self.assertIn(prediction, script)

    def test_lower_headroom_native_inference_defers_to_exact_save_terminal(self):
        marker = "candidate_refiner_v4_cell707_saved_20260823/SAVE_ONLY_COMPLETE.txt"
        self.assertIn(marker, self.union_native_oof)
        self.assertLess(
            self.union_native_oof.index("exact707_priority_terminal"),
            self.union_native_oof.index("wait_any_gpu"),
        )

    def test_prune_composition_is_label_isolated_and_reuses_primary_cache(self):
        script = self.prune_composition
        self.assertIn("FULL_VOLUME_OOF_COMPLETE.json", script)
        self.assertIn('work="${primary}/fold${fold}/work_bbox24"', script)
        self.assertIn('base="${staging}/fold${fold}/proposals/currentgf"', script)
        self.assertIn('prune_modes["currentgf"]="prune_base"', script)
        self.assertNotIn("--empty-base", script)
        self.assertNotIn("apply_candidate_refiner.py\" infer", script)
        self.assertNotIn("validation/Validation-Public", script)
        self.assertNotIn("Validation-Hidden", script)
        self.assertNotIn("package_codabench", script)

    def test_prune_composition_requires_two_paired_metric_promotions(self):
        script = self.prune_composition
        self.assertIn("prune_vs_currentgf.robust.json", script)
        self.assertIn("prune_vs_additive.robust.json", script)
        self.assertIn("--min-dsc-gain 0.002 --min-nsd-gain 0.002", script)
        self.assertIn('base["decision"]=="PROMOTE" and additive["decision"]=="PROMOTE"', script)

    def test_prune_public_save_is_symmetric_and_never_uploads(self):
        script = self.prune_chain
        self.assertIn('WEAK707_PUBLIC_BASE="${public_base}"', script)
        self.assertIn("WEAK707_PUBLIC_ALLOW_PRUNE_BASE=1", script)
        self.assertIn('WEAK707_SAVE_BASE="${hidden_base}"', script)
        self.assertIn("WEAK707_SAVE_ALLOW_PRUNE_BASE=1", script)
        self.assertIn("--expected-cases 50", script)
        self.assertIn("len(members)!=100", script)
        self.assertIn('"uploaded":False', script)
        self.assertNotIn("codabench", script.lower())

    def test_generic_deployment_rejects_unauthorized_empty_base_pruning(self):
        self.assertIn("WEAK707_PUBLIC_ALLOW_PRUNE_BASE", self.public)
        self.assertIn("deployments=(anchor)", self.public)
        self.assertIn("WEAK707_SAVE_ALLOW_PRUNE_BASE", self.save)
        self.assertIn('[[ "${deployment}" == anchor ]]', self.save)


if __name__ == "__main__":
    unittest.main()
