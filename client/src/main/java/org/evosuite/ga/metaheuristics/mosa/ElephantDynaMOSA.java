package org.evosuite.ga.metaheuristics.mosa;

import org.evosuite.Properties;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;
import org.evosuite.ga.comparators.OnlyCrowdingComparator;
import org.evosuite.ga.metaheuristics.mosa.structural.MultiCriteriaManager;
import org.evosuite.ga.operators.ranking.CrowdingDistance;
import org.evosuite.testcase.TestChromosome;
import org.evosuite.utils.LoggingUtils;
import org.evosuite.utils.Randomness;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Implementation of the ElephantMOSA
 * 
 */
public class ElephantDynaMOSA extends AbstractMOSA {

	private static final long serialVersionUID = -1648258301758721069L;

	private static final Logger logger = LoggerFactory.getLogger(ElephantDynaMOSA.class);

	/** Manager to determine the test goals to consider at each generation */
	protected MultiCriteriaManager goalsManager = null;

	protected CrowdingDistance<TestChromosome> distance = new CrowdingDistance<>();

	private List<List<TestChromosome>> clans = new ArrayList<>();

	/**
	 * Constructor based on the abstract class {@link AbstractMOSA}.
	 *
	 * @param factory
	 */
	public ElephantDynaMOSA(ChromosomeFactory<TestChromosome> factory) {
		super(factory);

		if (Properties.POPULATION < Properties.NUMBER_OF_ELEPHANT_CLANS) {
			LoggingUtils.getEvoLogger().warn(
					"Number of elephant clans cannot be bigger than population. Value adjusted to be equal to population");
			Properties.NUMBER_OF_ELEPHANT_CLANS = Properties.POPULATION;
		}
	}

	private void InitializeClans() {
		for (int i = 0; i < Properties.NUMBER_OF_ELEPHANT_CLANS; i++) {
			int index_start = i * population.size() / Properties.NUMBER_OF_ELEPHANT_CLANS;
			int index_end = (i + 1) * (population.size()) / Properties.NUMBER_OF_ELEPHANT_CLANS;

			List<TestChromosome> clan = new ArrayList<>();
			for (int j = index_start; j < index_end; j++) {
				clan.add(population.get(j));
			}
			clans.add(clan);
		}
	}
	
	protected List<TestChromosome> breedNextGenerationClan(List<TestChromosome> clan) {

		List<TestChromosome> offspringClan = new ArrayList<>(clan.size());
		// we apply only clan.size()/2 iterations since in each generation
		// we generate two offsprings
		TestChromosome matriarch = clan.get(0); //TODO percorrer clan, take best (fitness)
		for (int i = 0; i < clan.size() / 2 && !this.isFinished(); i++) {
			// select best individuals

			/*
			 * the same individual could be selected twice! Is this a problem for crossover?
			 * Because crossing over an individual with itself will most certainly give you
			 * the same individual again...
			 */

			
			TestChromosome parent2 = this.selectionFunction.select(clan);
			TestChromosome offspring1 = matriarch.clone();
			TestChromosome offspring2 = parent2.clone();
			// apply crossover
			if (Randomness.nextDouble() <= Properties.CROSSOVER_RATE) {
				try {
					this.crossoverFunction.crossOver(offspring1, offspring2);
				} catch (ConstructionFailedException e) {
					logger.debug("CrossOver failed.");
					continue;
				}
			}

			this.removeUnusedVariables(offspring1);
			this.removeUnusedVariables(offspring2);
			
			if (offspring1.isChanged()) {
				this.clearCachedResults(offspring1);
				offspring1.updateAge(this.currentIteration);
				this.calculateFitness(offspring1);			
			}

			if (offspring2.isChanged()) {
				this.clearCachedResults(offspring2);
				offspring2.updateAge(this.currentIteration);
				this.calculateFitness(offspring2);	
			}
			offspringClan.add(offspring1);
			offspringClan.add(offspring2);
		}
		notifyMutation(matriarch);
		matriarch.mutate();
		if (matriarch.isChanged()) {
			matriarch.updateAge(currentIteration);
		}
		offspringClan.add(matriarch);
		
		logger.info("Number of clan offsprings = {}", offspringClan.size());
		return offspringClan;
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		for (int i = 0; i < Properties.NUMBER_OF_ELEPHANT_CLANS; i++) {
			// Generate offspring, compute their fitness, update the archive and coverage
			// goals.
			List<TestChromosome> offspringClan = this.breedNextGenerationClan(clans.get(i));

			// Create the union of parents and offspring
			List<TestChromosome> union = new ArrayList<>(clans.get(i).size() + offspringClan.size());
			union.addAll(clans.get(i));
			union.addAll(offspringClan);

			// Ranking the union
			logger.debug("Union Size = {}", union.size());

			// Ranking the union using the best rank algorithm (modified version of the non
			// dominated
			// sorting algorithm)
			this.rankingFunction.computeRankingAssignment(union, this.goalsManager.getCurrentGoals());

			// let's form the next population using "preference sorting and non-dominated
			// sorting" on the
			// updated set of goals
			int remain = Math.max(clans.get(i).size(), this.rankingFunction.getSubfront(0).size());
			int index = 0;
			clans.get(i).clear();

			// Obtain the first front
			List<TestChromosome> front = this.rankingFunction.getSubfront(index);

			// Successively iterate through the fronts (starting with the first
			// non-dominated front)
			// and insert their members into the population for the next generation. This is
			// done until
			// all fronts have been processed or we hit a front that is too big to fit into
			// the next
			// population as a whole.
			while ((remain > 0) && (remain >= front.size()) && !front.isEmpty()) {
				// Assign crowding distance to individuals
				this.distance.fastEpsilonDominanceAssignment(front, this.goalsManager.getCurrentGoals());

				// Add the individuals of this front
				clans.get(i).addAll(front);

				// Decrement remain
				remain = remain - front.size();

				// Obtain the next front
				index++;
				if (remain > 0) {
					front = this.rankingFunction.getSubfront(index);
				}
			}
			

			// In case the population for the next generation has not been filled up
			// completely yet,
			// we insert the best individuals from the current front (the one that was too
			// big to fit
			// entirely) until there are no more free places left. To this end, and in an
			// effort to
			// promote diversity, we consider those individuals with a higher crowding
			// distance as
			// being better.
			if (remain > 0 && !front.isEmpty()) { // front contains individuals to insert
				this.distance.fastEpsilonDominanceAssignment(front, this.goalsManager.getCurrentGoals());
				front.sort(new OnlyCrowdingComparator<>());
				for (int k = 0; k < remain; k++) {
					clans.get(i).add(front.get(k));
				}
			}
			
			clans.set(i, clans.get(i).subList(0, clans.get(i).size()-Properties.NUMBER_OF_MALE_ELEPHANTS_PER_CLAN));
			// male separation
			for (int j = 0; j < Properties.NUMBER_OF_MALE_ELEPHANTS_PER_CLAN; j++) {
				// eq male elephant
				TestChromosome newElephant;
				if (this.getCoveredGoals().size() == 0 || Randomness.nextBoolean() || !Properties.SELECT_NEW_ELEPHANTS_FROM_ARCHIVE) {
					newElephant = chromosomeFactory.getChromosome();
				} else {
					newElephant = Randomness.choice(this.getSolutions()).clone();
					newElephant.mutate();
				}
				if(newElephant.isChanged()) {
					fitnessFunctions.forEach(newElephant::addFitness);
					newElephant.updateAge(currentIteration);
					calculateFitness(newElephant);	
				}
				clans.get(i).add(newElephant);
			}
		}
		// join clans to form population
		for (List<TestChromosome> c : clans) {
			population.addAll(c);
		}
		this.currentIteration++;
		// logger.debug("N. fronts = {}", ranking.getNumberOfSubfronts());
		// logger.debug("1* front size = {}", ranking.getSubfront(0).size());
		logger.debug("Covered goals = {}", goalsManager.getCoveredGoals().size());
		logger.debug("Current goals = {}", goalsManager.getCurrentGoals().size());
		logger.debug("Uncovered goals = {}", goalsManager.getUncoveredGoals().size());
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void generateSolution() {
		logger.debug("executing generateSolution function");

		// Set up the targets to cover, which are initially free of any control
		// dependencies.
		// We are trying to optimize for multiple targets at the same time.
		this.goalsManager = new MultiCriteriaManager(this.fitnessFunctions);

		LoggingUtils.getEvoLogger().info("* Initial Number of Goals in DynaMOSA = "
				+ this.goalsManager.getCurrentGoals().size() + " / " + this.getUncoveredGoals().size());

		logger.debug("Initial Number of Goals = " + this.goalsManager.getCurrentGoals().size());

		if (this.population.isEmpty()) {
			// Initialize the population by creating solutions at random.
			this.initializePopulation();
			InitializeClans();
		}

		// Calculate dominance ranks and crowding distance. This is required to decide
		// which
		// individuals should be used for mutation and crossover in the first iteration
		// of the main
		// search loop.
		for (int i = 0; i < Properties.NUMBER_OF_ELEPHANT_CLANS; i++) {
			this.rankingFunction.computeRankingAssignment(clans.get(i), this.goalsManager.getCurrentGoals());
			for (int j = 0; j < this.rankingFunction.getNumberOfSubfronts(); j++) {
				this.distance.fastEpsilonDominanceAssignment(this.rankingFunction.getSubfront(j),
						this.goalsManager.getCurrentGoals());
			}
		}
		// Evolve the population generation by generation until all goals have been
		// covered or the
		// search budget has been consumed.
		while (!isFinished() && this.goalsManager.getUncoveredGoals().size() > 0) {
			this.evolve();
			this.notifyIteration();
		}

		this.notifySearchFinished();
	}

	/**
	 * Calculates the fitness for the given individual. Also updates the list of
	 * targets to cover, as well as the population of best solutions in the archive.
	 *
	 * @param c the chromosome whose fitness to compute
	 */
	@Override
	protected void calculateFitness(TestChromosome c) {
		if (!isFinished()) {
			// this also updates the archive and the targets
			this.goalsManager.calculateFitness(c, this);
			this.notifyEvaluation(c);
		}
	}
}
