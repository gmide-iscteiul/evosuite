package org.evosuite.ga.metaheuristics;

import java.util.ArrayList;
import java.util.List;

import org.evosuite.Properties;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;
import org.evosuite.utils.LoggingUtils;

/**
 * ElephantHerdingOptimization implementation
 *
 * @author
 */
public class ElephantHerdingOptimization<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	private static final long serialVersionUID = -8342568601726499012L;
	private final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(ElephantHerdingOptimization.class);
	private List<List<T>> clans = new ArrayList<>();

	/**
	 * Constructor
	 *
	 * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public ElephantHerdingOptimization(ChromosomeFactory<T> factory) {
		super(factory);

		if (Properties.POPULATION < Properties.NUMBER_OF_ELEPHANT_CLANS) {
			LoggingUtils.getEvoLogger().warn(
					"Number of elephant clans cannot be bigger than population. Value adjusted to be equal to population");
			Properties.NUMBER_OF_ELEPHANT_CLANS = Properties.POPULATION;
		}
	}

	private void sortClans(List<List<T>> list) {
		for (List<T> c : list) {
			sortPopulation(c);
		}
	}

	private void calculateFitnessAndSortClans(List<List<T>> list) {
		for (List<T> c : list) {
			calculateFitnessAndSortPopulation(c);
		}
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>();
		List<List<T>> newGenerationClans = new ArrayList<>();
		/*
		 * population -> group of clans -> group of elephants population = nº clans * nº
		 * elephant per clan male leave clan (worst solutions) matriarch (best of each
		 * clan)
		 */

		// update clan phase
		for (int i = 0; i < Properties.NUMBER_OF_ELEPHANT_CLANS; i++) {
			// eq for matriarch of each clan
			List<T> clan = new ArrayList<>();
			T elephant_matriarch = clans.get(i).get(0);

			for (int j = 1; j < clans.get(i).size(); j++) {
				// eq for generic elephant
				T elephant = clans.get(i).get(j);
				try {
					crossoverFunction.crossOver(elephant, elephant_matriarch.clone());
					if (elephant.isChanged()) {
						elephant.updateAge(currentIteration);
					}
				} catch (ConstructionFailedException e) {
					logger.info("Crossover failed.");
				} finally {
					clan.add(elephant);
				}
			}
			notifyMutation(elephant_matriarch);
			elephant_matriarch.mutate();
			if (elephant_matriarch.isChanged()) {
				elephant_matriarch.updateAge(currentIteration);
			}
			clan.add(elephant_matriarch);
			newGenerationClans.add(clan);
		}

		// Determine Fitness
		calculateFitnessAndSortClans(newGenerationClans);

		// male separation
		for (int i = 0; i < Properties.NUMBER_OF_ELEPHANT_CLANS; i++) {
			int numberOfMaleElephants = 0;
			for (int j = (clans.get(i).size() - 1); j > 0; j--) {
				// eq male elephant
				T male_elephant = newGenerationClans.get(i).get(j);
				newGenerationClans.get(i).remove(male_elephant);
				T newElephant = chromosomeFactory.getChromosome();
				fitnessFunctions.forEach(newElephant::addFitness);
				newElephant.updateAge(currentIteration);
				calculateFitness(newElephant);
				newGenerationClans.get(i).add(newElephant);
				numberOfMaleElephants++;
				if (numberOfMaleElephants < Properties.NUMBER_OF_MALE_ELEPHANTS_PER_CLAN) {
					break;
				}
			}
		}

		// join clans to form population
		for (List<T> c : newGenerationClans) {
			newGeneration.addAll(c);
		}

		population = newGeneration;
		clans = newGenerationClans;
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

		// Set up elephant clans
		for (int i = 0; i < Properties.NUMBER_OF_ELEPHANT_CLANS; i++) {
			int index_start = i * population.size() / Properties.NUMBER_OF_ELEPHANT_CLANS;
			int index_end = (i + 1) * (population.size()) / Properties.NUMBER_OF_ELEPHANT_CLANS;

			List<T> clan = new ArrayList<>();
			for (int j = index_start; j < index_end; j++) {
				clan.add(population.get(j));
			}
			clans.add(clan);
		}
		// Determine fitness
		calculateFitnessAndSortPopulation();
		sortClans(clans);
		this.notifyIteration();
	}

	/** {@inheritDoc} */
	@Override
	public void generateSolution() {
		if (Properties.ENABLE_SECONDARY_OBJECTIVE_AFTER > 0 || Properties.ENABLE_SECONDARY_OBJECTIVE_STARVATION) {
			disableFirstSecondaryCriterion();
		}
		if (population.isEmpty())
			initializePopulation();

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
			sortClans(clans);
			
			// Local Search
			// applyLocalSearch();

//			double newFitness = getBestIndividual().getFitness();
//
//			if (getFitnessFunction().isMaximizationFunction())
//				assert (newFitness >= bestFitness)
//						: "best fitness was: " + bestFitness + ", now best fitness is " + newFitness;
//			else
//				assert (newFitness <= bestFitness)
//						: "best fitness was: " + bestFitness + ", now best fitness is " + newFitness;
//			bestFitness = newFitness;

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
		updateBestIndividualFromArchive();
		notifySearchFinished();
	}
}