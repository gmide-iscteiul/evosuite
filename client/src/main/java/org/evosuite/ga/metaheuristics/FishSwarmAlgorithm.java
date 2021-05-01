package org.evosuite.ga.metaheuristics;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.evosuite.Properties;
import org.evosuite.TimeController;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;
import org.evosuite.utils.Randomness;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * FishSwarmAlgorithm implementation
 *
 * @author
 */
public class FishSwarmAlgorithm<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	private static final long serialVersionUID = -4220392757546401915L;
	private final Logger logger = LoggerFactory.getLogger(FishSwarmAlgorithm.class);
	private List<Integer> neighbourhood = new ArrayList<>();

	/**
	 * Constructor
	 *
	 * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public FishSwarmAlgorithm(ChromosomeFactory<T> factory) {
		super(factory);
	}

	private T SwarmPhase(T fish, int indexNumber) {
		boolean condition = false;
		T middleNeighbour = null;
		if (!neighbourhood.isEmpty()) {
			int middleNeighbourIndex = neighbourhood.get(neighbourhood.size() / 2);
			middleNeighbour = population.get(middleNeighbourIndex).clone();

			double neighbourDivideByNeighbourhood = middleNeighbour.getFitness() / neighbourhood.size();
			double fishTimesConcentration = fish.getFitness() * Properties.FISH_CONCENTRATION;

			if (getFitnessFunction().isMaximizationFunction()) {
				if (neighbourDivideByNeighbourhood > fishTimesConcentration) {
					condition = true;
				}
			} else {
				if (neighbourDivideByNeighbourhood < fishTimesConcentration) {
					condition = true;
				}
			}
		}
		if (condition) {
			try {
				crossoverFunction.crossOver(fish, middleNeighbour);
			} catch (ConstructionFailedException e) {
				logger.info("Crossover failed.");
				fish = population.get(indexNumber);
			}
		} else {
			preyPhase(fish, indexNumber);
		}
		return fish;
	}

	private T FollowPhase(T fish, int indexNumber) {
		boolean condition = false;
		T bestNeighbour = null;
		if (!neighbourhood.isEmpty()) {
			int bestNeighbourIndex = neighbourhood.get(0);
			bestNeighbour = population.get(bestNeighbourIndex).clone();

			double neighbourDivideByNeighbourhood = bestNeighbour.getFitness() / neighbourhood.size();
			double fishTimesConcentration = fish.getFitness() * Properties.FISH_CONCENTRATION;

			if (getFitnessFunction().isMaximizationFunction()) {
				if (neighbourDivideByNeighbourhood > fishTimesConcentration) {
					condition = true;
				}
			} else {
				if (neighbourDivideByNeighbourhood < fishTimesConcentration) {
					condition = true;
				}
			}
		}
		if (condition) {
			try {
				crossoverFunction.crossOver(fish, bestNeighbour);
			} catch (ConstructionFailedException e) {
				logger.info("Crossover failed.");
				fish = population.get(indexNumber);
			}
		} else {
			preyPhase(fish, indexNumber);
		}
		return fish;
	}

	private void preyPhase(T fish, int indexNumber) {
		if (!neighbourhood.isEmpty()) {
			for (int i = 0; i < Properties.NUMBER_OF_ATTEMPTS; i++) {
				int randomNumber = Randomness.nextInt(neighbourhood.size());
				int randomNeighbourIndex = neighbourhood.get(randomNumber);

				if (randomNeighbourIndex < indexNumber) {
					T randomNeighbour = population.get(randomNeighbourIndex).clone();
					try {
						crossoverFunction.crossOver(fish, randomNeighbour);
					} catch (ConstructionFailedException e) {
						logger.info("Crossover failed.");
						fish = population.get(indexNumber);
					}
					return;
				}
			}
		}
		randomPhase(fish);
	}

	private void randomPhase(T fish) {
		notifyMutation(fish);
		fish.mutate();
	}

	private T bestFish(T fish1, T fish2) {
		if (getFitnessFunction().isMaximizationFunction()) {
			return (fish1.getFitness() > fish2.getFitness()) ? fish1 : fish2;
		} else {
			return (fish1.getFitness() < fish2.getFitness()) ? fish1 : fish2;
		}
	}

	private void createNeighbourhood(int index) {

		double highThreshhold = population.get(index).getFitness()
				+ (population.get(index).getFitness() * Properties.FISH_NEIGHBOURHOOD);
		double lowThreshhold = population.get(index).getFitness()
				- (population.get(index).getFitness() * Properties.FISH_NEIGHBOURHOOD);

		// find better neighbours
		for (int i = index - 1; i >= 0; i--) {
			double neighbourFitness = population.get(i).getFitness();

			if (getFitnessFunction().isMaximizationFunction()) {
				if (highThreshhold >= neighbourFitness) {
					neighbourhood.add(i);
				} else {
					break;
				}
			} else {
				if (lowThreshhold <= neighbourFitness) {
					neighbourhood.add(i);
				} else {
					break;
				}
			}
		}

		// sort neighbourhood
		if (neighbourhood.size() > 1) {
			Collections.sort(neighbourhood);
		}
		

		// find worst neighbours
		for (int i = index + 1; i < population.size(); i++) {
			double neighbourFitness = population.get(i).getFitness();

			if (getFitnessFunction().isMaximizationFunction()) {
				if (lowThreshhold <= neighbourFitness) {
					neighbourhood.add(i);
				} else {
					break;
				}
			} else {
				if (highThreshhold >= neighbourFitness) {
					neighbourhood.add(i);
				} else {
					break;
				}
			}
		}
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>(elitism());

		for (int i = newGeneration.size(); i < population.size(); i++) {
			T oldFish = population.get(i).clone();
			createNeighbourhood(i); // save the indexes of the neighbours

			T fish1 = SwarmPhase(oldFish, i);
			T fish2 = FollowPhase(oldFish, i);

			// bestBehaviourPhase
			T newFish = bestFish(fish1, fish2);
			if (newFish.isChanged()) {
				newFish.updateAge(currentIteration);
			}
			newGeneration.add(newFish);
			neighbourhood.clear();
		}

		population = newGeneration;
		// archive
		updateFitnessFunctionsAndValues();
		//
		currentIteration++;
	}

	/** {@inheritDoc} */
	@Override
	public void initializePopulation() {
		notifySearchStarted();
		currentIteration = 0;

		// Set up initial population
		generateInitialPopulation(Properties.POPULATION);
		// Determine fitness
		calculateFitnessAndSortPopulation();

		this.notifyIteration();
	}

	/** {@inheritDoc} */
	@Override
	public void generateSolution() {
		if (Properties.ENABLE_SECONDARY_OBJECTIVE_AFTER > 0 || Properties.ENABLE_SECONDARY_OBJECTIVE_STARVATION) {
			disableFirstSecondaryCriterion();
		}
		if (population.isEmpty()) {
			initializePopulation();
		}

		logger.debug("Starting evolution");
		int starvationCounter = 0;
		double bestFitness = Double.MAX_VALUE;
		double lastBestFitness = Double.MAX_VALUE;
		if (getFitnessFunction().isMaximizationFunction()) {
			bestFitness = 0.0;
			lastBestFitness = 0.0;
		}

		while (!isFinished()) {
			logger.debug("Current population: " + getAge() + "/" + Properties.SEARCH_BUDGET);
			logger.info("Best fitness: " + getBestIndividual().getFitness());
			evolve();
			// Determine fitness
			calculateFitnessAndSortPopulation();

			// Local Search
			// applyLocalSearch();

			double newFitness = getBestIndividual().getFitness();

			if (Properties.ELITE > 0) {
				if (getFitnessFunction().isMaximizationFunction())
					assert (newFitness >= bestFitness)
							: "best fitness was: " + bestFitness + ", now best fitness is " + newFitness;
				else
					assert (newFitness <= bestFitness)
							: "best fitness was: " + bestFitness + ", now best fitness is " + newFitness;
				bestFitness = newFitness;
			}

			if (Double.compare(bestFitness, lastBestFitness) == 0) {
				starvationCounter++;
			} else {
				logger.info("reset starvationCounter after " + starvationCounter + " iterations");
				starvationCounter = 0;
				lastBestFitness = bestFitness;

			}

			updateSecondaryCriterion(starvationCounter);

			this.notifyIteration();
		}
		TimeController.execute(this::updateBestIndividualFromArchive, "update from archive", 5_000);
		notifySearchFinished();
	}
}