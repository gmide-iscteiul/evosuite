package org.evosuite.ga.metaheuristics;

import java.util.ArrayList;
import java.util.List;

import org.evosuite.Properties;
import org.evosuite.TimeController;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;
import org.evosuite.utils.LoggingUtils;
import org.evosuite.utils.Randomness;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ChickenSwarmOptimization implementation
 *
 * @author
 */
public class ChickenSwarmOptimization<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	private static final long serialVersionUID = -6510209377016867332L;
	private final Logger logger = LoggerFactory.getLogger(ChickenSwarmOptimization.class);
	private List<List<T>> groups = new ArrayList<>();
	private List<T> chicks = new ArrayList<>();
	private List<Integer> mothers = new ArrayList<>();

	/**
	 * Constructor
	 *
	 * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public ChickenSwarmOptimization(ChromosomeFactory<T> factory) {
		super(factory);
		if (Properties.POPULATION < Properties.NUMBER_OF_ROOSTERS + Properties.NUMBER_OF_CHICKS + Properties.NUMBER_OF_MOTHER_HENS) {
			LoggingUtils.getEvoLogger().warn(
					"Population must at least have the same number as the sum of roosters, chicks and mother hens. Value adjusted to that sum");
			Properties.POPULATION = Properties.NUMBER_OF_ROOSTERS + Properties.NUMBER_OF_CHICKS + Properties.NUMBER_OF_MOTHER_HENS;
		}
		if (Properties.NUMBER_OF_CHICKS < Properties.NUMBER_OF_MOTHER_HENS) {
			LoggingUtils.getEvoLogger().warn(
					"Number of chicks must at minimum be the same as the number of mother hens.  Value adjusted be the same as the number of mother hens");
			Properties.NUMBER_OF_CHICKS = Properties.NUMBER_OF_MOTHER_HENS;
		}
	}
	
	private void UpdateChickenSwarm() {
		
		// decide swarm' ranks
		groups = new ArrayList<>();
		for (int i = 0; i < Properties.NUMBER_OF_ROOSTERS; i++) {
			List<T> g = new ArrayList<>();
			T rooster = population.get(i);
			g.add(rooster);
			groups.add(g);
		}
		for (int i = Properties.NUMBER_OF_ROOSTERS; i < population.size() - Properties.NUMBER_OF_CHICKS; i++) {
			T hen = population.get(i);
			int randomGroup = Randomness.nextInt(Properties.NUMBER_OF_ROOSTERS);
			groups.get(randomGroup).add(hen);
		}
		chicks = new ArrayList<>();
		for (int i = population.size() - Properties.NUMBER_OF_CHICKS; i < population.size(); i++) {
			T chick = population.get(i);
			chicks.add(chick);
		}	
		
		// decide mother-child pairs
		mothers = new ArrayList<>();	
		for (int i = 0; i < Properties.NUMBER_OF_MOTHER_HENS; i++) {
			int randomGroup = Randomness.nextInt(Properties.NUMBER_OF_ROOSTERS);
			if(groups.get(randomGroup).size() > 1) { // if there's hen in the group
				int randomHen = Randomness.nextInt(1, groups.get(randomGroup).size());
				mothers.add(randomGroup);
				mothers.add(randomHen);
			} else {
				i--;
			}
		}
			

//		// decide mother-child pairs
//		for (int i = 0; i < Properties.NUMBER_OF_MOTHER_HENS; i++) {
//			int r = Randomness.nextInt(Properties.NUMBER_OF_ROOSTERS, population.size() - Properties.NUMBER_OF_CHICKS);
//			T mother = population.get(r);
//			henChickPairs.add(mother);
//		}
//		for (int i = population.size() - Properties.NUMBER_OF_CHICKS; i < population.size(); i++) {
//			T chick = population.get(i);
//			henChickPairs.add(chick);
//		}		
	}


	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>();
		List<List<T>> newGenerationGroups = new ArrayList<>();
		List<T> newGenerationChicks = new ArrayList<>();

		if(currentIteration % Properties.CHICKEN_SWARM_UPDATE_INTERVAL == 0) {
			UpdateChickenSwarm();
		}
		
		
		for (int i = 0; i < groups.size(); i++) {	
			List<T> newGroup = new ArrayList<>();			
			T rooster = groups.get(i).get(0).clone();
			newGroup.add(rooster);
			// hen phase
			for (int j = 1; j < groups.get(i).size(); j++) { 
				T hen = groups.get(i).get(j).clone();
				try {
					crossoverFunction.crossOver(hen, rooster.clone());
					if (groups.get(i).size() > 2) { // if there's more than 1 hen in the group
						int randomHenIndex = Randomness.nextInt(1, groups.get(i).size());
						if (randomHenIndex == j) { // to guarantee crossover with another hen
							if (randomHenIndex != 1) {
								randomHenIndex--;
							} else {
								randomHenIndex++;
							}
						}
						T randomHen = groups.get(i).get(randomHenIndex).clone();
						crossoverFunction.crossOver(hen, randomHen);
					}
					if (hen.isChanged()) {
						hen.updateAge(currentIteration);
					}
					newGroup.add(hen);
				} catch (ConstructionFailedException e) {
					logger.info("Crossover/Mutation failed.");
					newGroup.add(groups.get(i).get(j));
				}
			}
			// rooster phase
			notifyMutation(rooster);
			rooster.mutate();
			if (rooster.isChanged()) {
				rooster.updateAge(currentIteration);
			}
			newGenerationGroups.add(newGroup);
		}
		
		
		// chick phase
		int j=0;
		for (int i = 0; i < Properties.NUMBER_OF_CHICKS; i++) {
			T chick = chicks.get(i).clone();
			T mother = groups.get(mothers.get(j)).get(mothers.get(++j));
			j++;
			try {
				crossoverFunction.crossOver(chick, mother);
				if (chick.isChanged()) {
					chick.updateAge(currentIteration);
				}
				newGenerationChicks.add(chick);
			} catch (ConstructionFailedException e) {
				logger.info("Crossover failed.");
				newGenerationChicks.add(chicks.get(i));
			}
		}
				
//		// chick phase
//		for (int i = 0; i < Properties.NUMBER_OF_CHICKS; i++) { 
//			T chick = henChickPairs.get(i + Properties.NUMBER_OF_MOTHER_HENS);
//			T mother = henChickPairs.get(i % Properties.NUMBER_OF_MOTHER_HENS).clone();
//			try {
//				crossoverFunction.crossOver(chick, mother.clone());
//			} catch (ConstructionFailedException e) {
//				logger.info("Crossover/Mutation failed.");
//			} finally {
//				newGeneration.add(chick);
//			}		
//		}
		
		// join groups to form population
		for (List<T> group : newGenerationGroups) {
			newGeneration.addAll(group);
		}
		
		chicks = newGenerationChicks;
		groups = newGenerationGroups;
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

			// remove Local Search
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
		TimeController.execute(this::updateBestIndividualFromArchive, "update from archive", 5_000);
		notifySearchFinished();
	}
}