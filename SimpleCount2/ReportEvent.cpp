#include "ReportEvent.h"

#include <sstream>


std::string eventTypeStr(EventType et) {
	switch (et) {
	case START:
		return "START";
	case END:
		return "END";
	default:
		return "(unknown)";
	}
}

std::string durationMeasureUnitStr(DurationsMeasureUnit mu) {
	switch (mu) {
	case SECONDS:
		return "s";
	case MILLISECONDS:
		return "ms";
	case MICROSECONDS:
		return "us";
	default:
		return "(unknown)";
	}
}

std::string ReportEvent::str() {
	
	std::ostringstream os;
	os << getEventKey() << "\t" << eventTypeStr(getEventType());
	return os.str();
}

std::string DurationDecoratorRecord::str() {

	std::ostringstream os;
	os << this->reportEvent.str() << 
		"\tDuration=" << 
		getEventDuration() <<
		durationMeasureUnitStr(getDurationMeasureUnit());
	return os.str();
}