package org.evosuite.statistics;

import java.util.ArrayList;
import java.util.List;

import org.evosuite.Properties;
import org.evosuite.TimeController;

public class StringSequenceOutputVariableFactory {
	protected RuntimeVariable variable;

	protected List<Long> timeStamps = new ArrayList<>();
	protected List<String> values = new ArrayList<>();

	private long startTime = 0L;

	protected String value;

	public StringSequenceOutputVariableFactory(RuntimeVariable variable, String startValue) {
		this.variable = variable;
		this.value = startValue;
	}

	protected String getValue() {
		return value;
	}

	public void update() {
		timeStamps.add(System.currentTimeMillis() - startTime);
		values.add(getValue());
	}

	public void setStartTime(long time) {
		this.startTime = time;
	}

	public void setValue(String value) {
		this.value = value;
	}

	public void setValue(Object value) {
		if (String.class.isInstance(value)) {
			this.setValue(value.toString());
		} else {
			throw new IllegalArgumentException("value of type " + value.getClass().getName()
					+ " is incompatible with expected type " + String.class.getName());
		}
	}

	public List<String> getVariableNames() {
		List<String> variables = new ArrayList<>();

		for (String suffix : getTimelineHeaderSuffixes()) {
			variables.add(variable.name() + suffix);
		}

		return variables;

	}

	public List<OutputVariable<String>> getOutputVariables() {
		List<OutputVariable<String>> variables = new ArrayList<>();

		for (String variableName : getVariableNames()) {
			OutputVariable<String> variable = new OutputVariable<>(variableName, getTimeLineValue(variableName));
			variables.add(variable);
		}

		return variables;
	}

	private String getTimeLineValue(String name) {
		long interval = Properties.TIMELINE_INTERVAL;

		int index = Integer.parseInt((name.split("_T"))[1]);
		long preferredTime = interval * index;

		/*
		 * No data. Is it even possible? Maybe if population is too large, and budget
		 * was not enough to get even first generation
		 */
		if (timeStamps.isEmpty()) {
			return "0"; // FIXXME - what else?
		}

		for (int i = 0; i < timeStamps.size(); i++) {
			/*
			 * find the first stamp that is after the time we would like to get coverage
			 * from
			 */
			long stamp = timeStamps.get(i);
			if (stamp < preferredTime) {
				continue;
			}

			if (i == 0) {
				/*
				 * it is the first element, so not much to do, we just use it as value
				 */
				return values.get(i);
			}

			/*
			 * If we do not want to interpolate, return last observed value
			 */
			if (!Properties.TIMELINE_INTERPOLATION) {
				return values.get(i - 1);
			}

			/*
			 * Now we interpolate the coverage, as usually we don't have the value for exact
			 * time we want
			 */
			long timeDelta = timeStamps.get(i) - timeStamps.get(i - 1);

			if (timeDelta > 0) {
				String[] array1 = values.get(i).split(";");
				String[] array2 = values.get(i - 1).split(";");

				String returnValue = "";

				int min = (array1.length < array2.length) ? array1.length : array2.length;

				for (int j = 0; j < min; j++) {
					double covDelta = Double.parseDouble(array1[j]) - Double.parseDouble(array2[j]);
					double ratio = covDelta / timeDelta;

					long diff = preferredTime - timeStamps.get(i - 1);
					Double cov = Double.parseDouble(array2[j]) + (diff * ratio);
					if (j == 0) {
						returnValue = String.valueOf(cov);
					} else {
						returnValue = String.join(";", returnValue, String.valueOf(cov));
					}
				}

				if (array1.length > min) {
					for (int j = min + 1; j < array1.length; j++) {
						returnValue = String.join(";", returnValue, String.valueOf(Double.parseDouble(array1[j])));
					}
				} else if (array2.length > min) {
					for (int j = min + 1; j < array2.length; j++) {
						returnValue = String.join(";", returnValue, String.valueOf(Double.parseDouble(array2[j])));
					}
				}
				// return values.get(i);
				return returnValue;

			}
		}

		/*
		 * No time stamp was higher. This might happen if coverage is 100% and we stop
		 * search. So just return last value seen
		 */

		return values.get(values.size() - 1);
	}

	private String[] getTimelineHeaderSuffixes() {
		int numberOfIntervals = calculateNumberOfIntervals();
		String[] suffixes = new String[numberOfIntervals];
		for (int i = 0; i < suffixes.length; i++) {
			/*
			 * NOTE: we start from T1 and not T0 because, by definition, coverage at T0 is
			 * equal to T0, and no point in showing it in a graph
			 */
			suffixes[i] = "_T" + (i + 1);
		}
		return suffixes;
	}

	private int calculateNumberOfIntervals() {
		long interval = Properties.TIMELINE_INTERVAL;
		/*
		 * We cannot just look at the obtained history, because the search might have
		 * finished earlier, eg if 100% coverage
		 */
		long totalTime = TimeController.getSearchBudgetInSeconds() * 1000L;

		int numberOfIntervals = (int) (totalTime / interval);
		return numberOfIntervals;
	}

	public static StringSequenceOutputVariableFactory getString(RuntimeVariable variable) {
		return new StringSequenceOutputVariableFactory(variable, "0.0");
	}

}
