package org.evosuite.ga.metaheuristics;

import java.util.ArrayList;
import java.util.List;

import org.evosuite.EvoSuite;
import org.evosuite.Properties;
import org.evosuite.SystemTestBase;
import org.evosuite.Properties.Algorithm;
import org.evosuite.Properties.Criterion;
import org.evosuite.Properties.SelectionFunction;
import org.evosuite.Properties.StoppingCondition;
import org.evosuite.ga.Chromosome;
import org.junit.Assert;
import org.junit.Test;

import com.examples.with.different.packagename.ClassHierarchyIncludingInterfaces;
import com.examples.with.different.packagename.XMLElement2;

public class MothFlameOptimizationSystemTest extends SystemTestBase {
	public List<Chromosome> setup(StoppingCondition sc, int budget, String cut) {
		Properties.CRITERION = new Criterion[1];
		Properties.CRITERION[0] = Criterion.BRANCH;
		Properties.ALGORITHM = Algorithm.MOTH_FLAME_OPTIMIZATION;
		Properties.POPULATION = 25;
		Properties.STOPPING_CONDITION = sc;
		Properties.SEARCH_BUDGET = budget;

		EvoSuite evosuite = new EvoSuite();

		String targetClass = cut;
		Properties.TARGET_CLASS = targetClass;

		String[] command = new String[] { "-generateSuite", "-class", targetClass };

		Object result = evosuite.parseCommandLine(command);
		Assert.assertNotNull(result);

		GeneticAlgorithm<?> ga = getGAFromResult(result);

		List<Chromosome> population = new ArrayList<>(ga.getBestIndividuals());

		return population;
	}

	@Test
	public void testMothFlameOptimizationWithLimitedTime() {

		List<Chromosome> population = this.setup(StoppingCondition.MAXTIME, 15, XMLElement2.class.getCanonicalName());

		for (Chromosome p : population) {
			Assert.assertNotEquals(p.getCoverage(), 1.0);
		}
	}

	@Test
	public void testMothFlameOptimizationWithLimitedGenerations() {

		List<Chromosome> population = this.setup(StoppingCondition.MAXGENERATIONS, 10,
				ClassHierarchyIncludingInterfaces.class.getCanonicalName());

		for (Chromosome p : population) {
			Assert.assertNotEquals(p.getCoverage(), 1.0);
		}
	}
}
