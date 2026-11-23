import unittest

from algorithms.astar import astar, bfs
from algorithms.bayes import update
from algorithms.mdp import decide
from config import BASELINE, STRESS
from pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
  def test_clear_sensor_proceeds_to_exit_a_with_verified_posterior(self):
    result = run_pipeline(observation="clear")
    self.assertAlmostEqual(result.posterior, 0.0555555556)
    self.assertEqual(result.decision.action, "PROCEED")
    self.assertEqual(result.selected_exit, "Exit A")
    self.assertEqual(result.route.path[-1], BASELINE.exit_a)


  def test_smoke_alert_detours_to_exit_b_around_monitored_cell(self):
    result = run_pipeline(observation="alert")
    self.assertAlmostEqual(result.posterior, 0.6538461538)
    self.assertEqual(result.decision.action, "DETOUR")
    self.assertEqual(result.selected_exit, "Exit B")
    self.assertEqual(result.route.path[-1], BASELINE.exit_b)
    self.assertNotIn(BASELINE.monitored_corridor, result.route.path)
    self.assertIn(BASELINE.monitored_corridor, result.effective_walls)


  def test_noisy_sensor_policy_tracks_posterior_and_threshold(self):
    reliable_alarm = run_pipeline(observation="alert", true_positive=0.60, false_positive=0.40)
    uninformative_alarm = run_pipeline(observation="alert", true_positive=0.60, false_positive=0.50)
    self.assertGreater(reliable_alarm.posterior, reliable_alarm.decision.threshold)
    self.assertEqual(reliable_alarm.decision.action, "DETOUR")
    self.assertLess(uninformative_alarm.posterior, uninformative_alarm.decision.threshold)
    self.assertEqual(uninformative_alarm.decision.action, "PROCEED")
    self.assertAlmostEqual(uninformative_alarm.decision.threshold, 0.30)

  def test_heat_sensor_uses_smoke_posterior_as_its_prior(self):
    result = run_pipeline(observation="alert", heat_observation="clear")
    self.assertEqual(len(result.bayes), 2)
    self.assertAlmostEqual(result.bayes[1].prior, result.bayes[0].posterior)
    self.assertAlmostEqual(result.posterior, 0.3207547170)
    self.assertEqual(result.decision.action, "DETOUR")


  def test_bayes_prior_extremes(self):
    for prior in (0.0, 1.0):
      with self.subTest(prior=prior):
        self.assertAlmostEqual(update(prior, "alert", 0.85, 0.15).posterior, prior)


  def test_mdp_posterior_extremes_and_exact_boundary_choose_safe_tie(self):
    self.assertEqual(decide(0.0).action, "PROCEED")
    self.assertEqual(decide(1.0).action, "DETOUR")
    self.assertEqual(decide(0.30).action, "DETOUR")


  def test_astar_and_bfs_return_equal_optimal_costs_on_both_maps(self):
    for building in (BASELINE, STRESS):
        a = astar(building.size, building.start, building.exit_a, building.walls)
        b = bfs(building.size, building.start, building.exit_a, building.walls)
        self.assertIsNotNone(a.path)
        self.assertIsNotNone(b.path)
        self.assertEqual(a.cost, b.cost)
        self.assertEqual(a.cost, len(a.path) - 1)


  def test_search_handles_blocked_and_unreachable_exits_without_crashing(self):
    self.assertIsNone(astar(3, (0, 0), (2, 2), {(2, 2)}).path)
    # A full wall row disconnects the start from the goal.
    self.assertIsNone(bfs(3, (0, 0), (2, 2), {(1, 0), (1, 1), (1, 2)}).path)


  def test_invalid_probabilities_are_rejected(self):
    for prior in (-0.01, 1.01):
      with self.subTest(prior=prior), self.assertRaises(ValueError):
        update(prior, "alert", 0.85, 0.15)
