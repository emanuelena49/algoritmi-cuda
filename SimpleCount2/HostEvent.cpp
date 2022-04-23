#include "HostEvent.h"

#include <sstream>

std::string HostEvent::getTimePointStr() {

	std::time_t time = std::chrono::system_clock::to_time_t(getTimePoint());
	return std::ctime(&time);
}


std::string HostEvent::str() {

	std::ostringstream os;
	os << "[" << getTimePointStr() << "]\t" << reportEvent.str();

	return os.str();
}

double calculateDuration(HostEvent startEvent, HostEvent endEvent, DurationsMeasureUnit measureUnit) {

	double seconds = (double) (endEvent.getTimePoint() - startEvent.getTimePoint()).count();

	switch (measureUnit)
	{
	case SECONDS:
		return seconds;
		break;
	case MILLISECONDS:
		return seconds / 1000;
		break;
	case MICROSECONDS:
		return seconds / 1000000;
		break;
	default:
		return -1;
		break;
	}
}