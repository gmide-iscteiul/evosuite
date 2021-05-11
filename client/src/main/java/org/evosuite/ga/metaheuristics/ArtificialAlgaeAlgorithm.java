package org.evosuite.ga.metaheuristics;

import java.util.ArrayList;
import java.util.List;

import org.evosuite.Properties;
import org.evosuite.TimeController;
import org.evosuite.Properties.SelectionFunction;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;
import org.evosuite.testcase.TestChromosome;
import org.evosuite.testsuite.TestSuiteChromosome;
import org.evosuite.utils.LoggingUtils;
import org.evosuite.utils.Randomness;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ArtificialAlgaeAlgorithm implementation
 *
 * @author
 */
public class ArtificialAlgaeAlgorithm<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	private static final long serialVersionUID = 4982755939858913909L;
	private final Logger logger = LoggerFactory.getLogger(ArtificialAlgaeAlgorithm.class);

	/**
	 * Constructor
	 *
	 * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public ArtificialAlgaeAlgorithm(ChromosomeFactory<T> factory) {
		super(factory);

		if (Properties.SELECTION_FUNCTION != SelectionFunction.TOURNAMENT) {
			LoggingUtils.getEvoLogger().warn("Originally, Artificial Algae Algorithm was implemented with a '"
					+ SelectionFunction.TOURNAMENT.name() + "' selection function. You may want to consider using it.");
		}
	}

	private T bestAlgae(T algae1, T algae2) {
		if (getFitnessFunction().isMaximizationFunction()) {
			return (algae1.getFitness() > algae2.getFitness()) ? algae1 : algae2;
		} else {
			return (algae1.getFitness() < algae2.getFitness()) ? algae1 : algae2;
		}
	}

	private int getOldestAlgaeIndex(List<T> list) {
		int index = Properties.ELITE;
		for (int i = Properties.ELITE + 1; i < list.size(); i++) {
			if (list.get(index).getAge() >= list.get(i).getAge()) {
				index = i;
			}
		}
		return index;
	}

	/** {@inheritDoc} */
	@SuppressWarnings("unchecked")
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>(elitism());

		// evolutionary process phase
		boolean starvation = true;
		for (int i = newGeneration.size(); i < population.size(); i++) {
			T finalAlgae = population.get(i);
			T algae = finalAlgae.clone();
			// energy = directly proportional to position;
			double energy = Properties.MAX_INITIAL_ENERGY - i;
			
			while (energy > 0 && !isFinished()) {
				energy -= (Properties.ENERGY_LOSS_RATE * Properties.MAX_INITIAL_ENERGY);
				T selectedAlgae = selectionFunction.select(population).clone();

				try {
					crossoverFunction.crossOver(algae, selectedAlgae);
					calculateFitness(algae);
				} catch (ConstructionFailedException e) {
					logger.info("Crossover failed.");
					algae = finalAlgae;
				}

				T newAlgae = bestAlgae(algae, finalAlgae);
				if (newAlgae == algae) { // new solution is better
					finalAlgae = newAlgae;
					starvation = false;
				} else {
					energy -= (Properties.ENERGY_LOSS_RATE * Properties.MAX_INITIAL_ENERGY);
				}
				if (energy > 0) {
					algae = finalAlgae.clone();
				}
			}
			if (!starvation) { // if a better solution was found
				finalAlgae.updateAge(currentIteration);
				newGeneration.add(finalAlgae);
			} else {
				newGeneration.add(population.get(i));
			}
		}

		// reproduction phase
		sortPopulation(newGeneration);

		TestSuiteChromosome bestAlgae = (TestSuiteChromosome) newGeneration.get(0);
		int random1 = Randomness.nextInt(bestAlgae.size());
		TestChromosome randomTest = bestAlgae.getTestChromosome(random1);
		
		TestSuiteChromosome worstAlgae = (TestSuiteChromosome) newGeneration.get(newGeneration.size() - 1);
		int random2 = Randomness.nextInt(worstAlgae.size());
		worstAlgae.setTestChromosome(random2, randomTest);
		
		calculateFitness((T) worstAlgae); // all individuals with fitness calculated

		// Adaptation phase
		if (Randomness.nextDouble() < Properties.ADAPTATION_RATE) {
			T bestAlgaeClone = newGeneration.get(0).clone();
			int index = getOldestAlgaeIndex(newGeneration);
			T oldestAlgae = newGeneration.get(index).clone();
			try {
				crossoverFunction.crossOver(oldestAlgae, bestAlgaeClone);
				calculateFitness(oldestAlgae);
				newGeneration.remove(index);
				newGeneration.add(oldestAlgae);
				if (oldestAlgae.isChanged()) {
					oldestAlgae.updateAge(currentIteration);
				}
			} catch (ConstructionFailedException e) {
				logger.info("Crossover failed.");
			}
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

			sortPopulation();

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