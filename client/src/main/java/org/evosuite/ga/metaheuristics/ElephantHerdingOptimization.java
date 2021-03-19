package org.evosuite.ga.metaheuristics;

import java.util.ArrayList;
import java.util.List;

import org.evosuite.Properties;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;

/**
 * ElephantHerdingOptimization implementation
 *
 * @author
 */
public class ElephantHerdingOptimization<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	
	private static final long serialVersionUID = -6639358870017674241L;
	private final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(ElephantHerdingOptimization.class);

	/**
	 * Constructor
	 *
	 * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public ElephantHerdingOptimization(ChromosomeFactory<T> factory) {
		super(factory);
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>();

		/*
		population -> group of clans -> group of elephants
		population = nº clans * nº elephant per clan
		male leave clan (worst solutions)
		matriarch (best of each clan)	
		*/
		
		
//		"Original" EHO		
//		//update clan phase
//		for(int i=0; i<Properties.NUMBER_OF_ELEPHANT_CLANS; i++) {
//			for(int j=0; j<Properties.NUMBER_OF_ELEPHANT_PER_CLAN; j++) {
//				if(matriarch of clan i) {
//					//eq for best of each clan
//				} else {
//					//eq for generic elephant
//				}
//			}
//		}
//		//male separation
//		for(int i=0; i<Properties.NUMBER_OF_ELEPHANT_CLANS; i++) {
//			for(int j=0; j<Properties.NUMBER_OF_MALE_ELEPHANTS_PER_CLAN; j++) {
//				//eq male elephant
//			}
//		}
		
		// update clan phase
		for (int i = 0; i < Properties.NUMBER_OF_ELEPHANT_CLANS; i++) {
			// eq 2 - for best of each clan = (mutation)
			T elephant_matriarch = population.get(i).clone();
			notifyMutation(elephant_matriarch);
			elephant_matriarch.mutate();
			newGeneration.add(elephant_matriarch);
			int index_start = i * (population.size() - Properties.NUMBER_OF_ELEPHANT_CLANS)
					/ Properties.NUMBER_OF_ELEPHANT_CLANS + Properties.NUMBER_OF_ELEPHANT_CLANS;
			int index_end = (i + 1) * (population.size() - Properties.NUMBER_OF_ELEPHANT_CLANS)
					/ Properties.NUMBER_OF_ELEPHANT_CLANS + Properties.NUMBER_OF_ELEPHANT_CLANS;

			if (index_end > population.size()) {
				index_end = population.size();
			}
			for (int j = index_start; j < index_end; j++) {
				// eq 1 - for generic elephant = (crossover)
				T elephant = population.get(j).clone();
				try {
					crossoverFunction.crossOver(elephant, elephant_matriarch.clone());
				} catch (ConstructionFailedException e) {
					logger.info("Crossover failed.");
				} finally {
					newGeneration.add(elephant);
				}
			}
		}

		// male separation
		for (int j = 0; j < Properties.NUMBER_OF_MALE_ELEPHANTS_PER_CLAN * Properties.NUMBER_OF_ELEPHANT_CLANS; j++) {
			// eq 4 - male elephant = (replaced by a new individual)
			population.remove(population.size() - 1 - j);
			T newElephant = chromosomeFactory.getChromosome();
			fitnessFunctions.forEach(newElephant::addFitness);
			newGeneration.add(newElephant);
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
			// Determine fitness
			calculateFitnessAndSortPopulation();

			////// remove Local Search?
			// applyLocalSearch();

			double newFitness = getBestIndividual().getFitness();

			if (getFitnessFunction().isMaximizationFunction())
				assert (newFitness >= bestFitness)
						: "best fitness was: " + bestFitness + ", now best fitness is " + newFitness;
			else
				assert (newFitness <= bestFitness)
						: "best fitness was: " + bestFitness + ", now best fitness is " + newFitness;
			bestFitness = newFitness;

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