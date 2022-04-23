#include "IExecutionLog.h"

#include <sstream>

std::string IExecutionLog::str(DurationsMeasureUnit durationsMeasureUnit) {

	std::ostringstream os;

	for (ReportEvent e : getReport(durationsMeasureUnit)) {
		os << e.str() << "\n";
	}

	return os.str();
}