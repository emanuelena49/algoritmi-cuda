#include "IExecutionLog.h"

#include <sstream>

std::string IExecutionLog::str(DurationsMeasureUnit durationsMeasureUnit) {

	std::ostringstream os;

	for (IReportEvent e : getReport(durationsMeasureUnit)) {
		os << e.str() << "\n";
	}

	return os.str();
}