package org.evosuite.ga.metaheuristics;

import java.util.ArrayList;
import java.util.List;

import org.evosuite.Properties;
import org.evosuite.Properties.SelectionFunction;
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

	
	private static final long serialVersionUID = -6639358870017674241L;
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
	
	private void sortClans() {
		for (List<T> c : clans) {
			sortPopulation(c);
		}
	}
	
	private void calculateFitnessAndSortClans() {
		for (List<T> c : clans) {
			calculateFitnessAndSortPopulation(c);
		}
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>();
		List<List<T>> newGenerationClans = new ArrayList<>();
		/*
		population -> group of clans -> group of elephants
		population = nº clans * nº elephant per clan
		male leave clan (worst solutions)
		matriarch (best of each clan)	
		*/
		
		
		// update clan phase
		for (int i = 0; i < Properties.NUMBER_OF_ELEPHANT_CLANS; i++) {
			// eq for matriarch of each clan
			List<T> clan = new ArrayList<>();
			T elephant_matriarch;
			if(clans.get(i).get(0).equals(population.get(0))) {	//save best individual
				elephant_matriarch = population.get(0).clone();
				clan.add(elephant_matriarch);
			} else {
				elephant_matriarch = clans.get(i).get(0).clone();
				notifyMutation(elephant_matriarch);
				elephant_matriarch.mutate();
				clan.add(elephant_matriarch);
			}
			for (int j = 1; j < clans.get(i).size(); j++) {
				// eq for generic elephant
				T elephant = clans.get(i).get(j).clone();
				try {
					crossoverFunction.crossOver(elephant, elephant_matriarch.clone());
				} catch (ConstructionFailedException e) {
					logger.info("Crossover failed.");
				} finally {
					clan.add(elephant);
				}
			}
			newGenerationClans.add(clan);
		}
		
		//Determine Fitness
		calculateFitnessAndSortClans();
		
		// male separation
		for (int i = 0; i < Properties.NUMBER_OF_ELEPHANT_CLANS; i++) {
			for (int k = (clans.get(i).size() - 1), j = 0; k > 0
					&& j < Properties.NUMBER_OF_MALE_ELEPHANTS_PER_CLAN; k--, j++) {
				// eq male elephant
				T male_elephant = newGenerationClans.get(i).get(k);
				newGenerationClans.get(i).remove(male_elephant);
				T newElephant = chromosomeFactory.getChromosome();
				fitnessFunctions.forEach(newElephant::addFitness);
				calculateFitness(newElephant);
				newGenerationClans.get(i).add(newElephant);
			}
		}

		// join clans to form population
		for (List<T> c : newGenerationClans) {
			newGeneration.addAll(c);
		}
		

//		// update clan phase
//		for (int i = 0; i < Properties.NUMBER_OF_ELEPHANT_CLANS; i++) {
//			// eq 2 - for best of each clan = (mutation)
//			T elephant_matriarch = population.get(i).clone();
//			notifyMutation(elephant_matriarch);
//			elephant_matriarch.mutate();
//			newGeneration.add(elephant_matriarch);
//			int index_start = i * (population.size() - Properties.NUMBER_OF_ELEPHANT_CLANS)
//					/ Properties.NUMBER_OF_ELEPHANT_CLANS + Properties.NUMBER_OF_ELEPHANT_CLANS;
//			int index_end = (i + 1) * (population.size() - Properties.NUMBER_OF_ELEPHANT_CLANS)
//					/ Properties.NUMBER_OF_ELEPHANT_CLANS + Properties.NUMBER_OF_ELEPHANT_CLANS;
//
//			if (index_end > population.size()) {
//				index_end = population.size();
//			}
//			for (int j = index_start; j < index_end; j++) {
//				// eq 1 - for generic elephant = (crossover)
//				T elephant = population.get(j).clone();
//				try {
//					crossoverFunction.crossOver(elephant, elephant_matriarch.clone());
//				} catch (ConstructionFailedException e) {
//					logger.info("Crossover failed.");
//				} finally {
//					newGeneration.add(elephant);
//				}
//			}
//		}
//
//		// Determine fitness
//		calculateFitnessAndSortPopulation();
//		
//		// male separation 
//		ADD&REMOVE SAME ELEMENT!!!!!!!!!!!!!!!!!! -> added "i" to solve issue
//		for (int j = 0, i=population.size()-1; i>0, j < Properties.NUMBER_OF_MALE_ELEPHANTS_PER_CLAN * Properties.NUMBER_OF_ELEPHANT_CLANS;i--, j++) {
//			// eq 4 - male elephant = (replaced by a new individual)
//			newGeneration.remove(i);
//			T newElephant = chromosomeFactory.getChromosome();
//			fitnessFunctions.forEach(newElephant::addFitness);
//			calculateFitness(newElephant);
//			newGeneration.add(newElephant);
//		}
		
		

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
		
		//Set up elephant clans
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
		sortClans();
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
			sortClans();
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