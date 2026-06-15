# Experiments

Experiment A uses hidden physical modes and analytic latent optimism to show latent value inflation under high `N`.

Experiment B trains a compact RSSM-style PyTorch model with encoder, recurrent state, stochastic prior/posterior, decoder, reward head, and value head.

Experiment C stresses hidden-mode belief collapse across ambiguous modes.

Experiment D varies horizon `H` and selection budget `N`.

Experiment E compares raw value, individual repairs, combined repair, random selection, and oracle scoring.

Experiment F evaluates controlled and learned receding-horizon planning.

Experiment G ablates disjoint pilot-label budget for calibration.

Experiment H sweeps OOD hidden-mode regimes and records harmful, neutral, and helpful high-`N` regions.

Experiment I adds three lightweight Gymnasium toy-text stochastic benchmarks.

Experiment J adds a high-risk belief-intervention stress test. It checks whether posterior-prior drift and belief-collapse diagnostics recover selected-tail real utility without relying only on pilot labels.

Experiment K adds three standard Gymnasium classic-control tasks with actual environment rollouts. CartPole and Acrobot expose optimistic latent-tail failure with repair recovery; MountainCar is kept as a helpful-boundary case.
