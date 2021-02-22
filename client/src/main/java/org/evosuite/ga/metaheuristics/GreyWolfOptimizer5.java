package org.evosuite.ga.metaheuristics;

import java.util.ArrayList;
import java.util.List;

import org.evosuite.Properties;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;
import org.evosuite.utils.Randomness;


/**
 * GreyWolfOptimizer implementation
 *
 * @author 
 */
public class GreyWolfOptimizer5<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	private static final long serialVersionUID = -8811115659916973474L;
	//private static final long serialVersionUID = 5043503777821916152L;
	
	private final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(GreyWolfOptimizer5.class);
	
	/**
	 * Constructor
	 *
	 * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public GreyWolfOptimizer5(ChromosomeFactory<T> factory) {
		super(factory);
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>();
		
		//double a = 2.0-((double)currentIteration*(2.0/(double)Properties.SEARCH_BUDGET)); 
		double a=2*this.progress();	//unlike the above "a", this one increases during search.
		
		T alpha=population.get(0).clone();
		T beta=population.get(1).clone();
		T delta=population.get(2).clone();
		
		newGeneration.add(alpha);
		newGeneration.add(beta);
		newGeneration.add(delta);
		
		for(int i=3; i < population.size(); i++) {
			T wolf=population.get(i).clone();
			
			/* 
			 * A- exploration -> mutation || exploitation -> crossover
			 * C- can always happens mutation
			 * 0.40 || 0.80
			 */
			double A = 2 * a * Randomness.nextDouble() - a;// Equation (3.3)
			double C = 2 * Randomness.nextDouble(); // Equation (3.4)
			try {
				if (A > 0.10) {
					// crossover
					crossoverFunction.crossOver(wolf, alpha.clone());
					crossoverFunction.crossOver(wolf, beta.clone());
					crossoverFunction.crossOver(wolf, delta.clone());
				}
				if (A <= 0.10 || C <= 1.0) {
					// mutation
					notifyMutation(wolf);
					wolf.mutate();
				}
			} catch (ConstructionFailedException e) {
				logger.info("CrossOver/Mutation failed.");
				continue;
			}
			if (wolf.isChanged()) {
				wolf.updateAge(currentIteration);
			}
			newGeneration.add(wolf);	
		}
		population = newGeneration;
        //archive
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
		if (Properties.ENABLE_SECONDARY_OBJECTIVE_AFTER > 0
				|| Properties.ENABLE_SECONDARY_OBJECTIVE_STARVATION) {
			disableFirstSecondaryCriterion();
		}
		if (population.isEmpty())
			initializePopulation();

		logger.debug("Starting evolution");
		int starvationCounter = 0;
		double bestFitness = Double.MAX_VALUE;
		double lastBestFitness = Double.MAX_VALUE;
		if (getFitnessFunction().isMaximizationFunction()){
			bestFitness = 0.0;
			lastBestFitness = 0.0;
		} 
		
		while (!isFinished()) {
			logger.debug("Current population: " + getAge() + "/" + Properties.SEARCH_BUDGET);
			logger.info("Best fitness: " + getBestIndividual().getFitness());
			
			evolve();
			// Determine fitness
			calculateFitnessAndSortPopulation();
			
		//////remove Local Search?
			//applyLocalSearch();	
			
			double newFitness = getBestIndividual().getFitness();

			if (getFitnessFunction().isMaximizationFunction())
				assert (newFitness >= bestFitness) : "best fitness was: " + bestFitness
						+ ", now best fitness is " + newFitness;
			else
				assert (newFitness <= bestFitness) : "best fitness was: " + bestFitness
						+ ", now best fitness is " + newFitness;
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