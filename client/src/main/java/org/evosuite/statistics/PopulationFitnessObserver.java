package org.evosuite.statistics;

import org.evosuite.ga.Chromosome;
import org.evosuite.ga.metaheuristics.GeneticAlgorithm;
import org.evosuite.ga.metaheuristics.SearchListener;
import org.evosuite.rmi.ClientServices;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class PopulationFitnessObserver<T extends Chromosome<T>> implements SearchListener<T> {

	private static final long serialVersionUID = -5097930559577892754L;

	private static final Logger logger = LoggerFactory.getLogger(PopulationFitnessObserver.class);

	public PopulationFitnessObserver() {
		// empty default constructor
	}

	@SuppressWarnings("rawtypes")
	public PopulationFitnessObserver(PopulationFitnessObserver that) {
		// empty copy constructor
	}

	@Override
	public void iteration(GeneticAlgorithm<T> algorithm) {

		List<T> population = algorithm.getPopulation();
		double fitnessOfIndividual = population.get(0).getFitness();
		String fitnesses = String.valueOf(fitnessOfIndividual);

		for (int i = 1; i < population.size(); i++) {
			fitnessOfIndividual = population.get(i).getFitness();
			fitnesses = String.join(";", fitnesses, String.valueOf(fitnessOfIndividual));
		}
		logger.info("Population Fitness " + fitnesses);
		ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable.PopulationFitnessTimeline,
				fitnesses);

	}

	@Override
	public void searchStarted(GeneticAlgorithm<T> algorithm) {
		// TODO Auto-generated method stub

	}

	@Override
	public void searchFinished(GeneticAlgorithm<T> algorithm) {
		// TODO Auto-generated method stub

	}

	@Override
	public void fitnessEvaluation(T individual) {
		// TODO Auto-generated method stub

	}

	@Override
	public void modification(T individual) {
		// TODO Auto-generated method stub

	}
}
