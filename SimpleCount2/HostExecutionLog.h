
#pragma once
#include "IExecutionLog.h"

#include "HostEvent.h"

#include <list>



class HostExecutionLog : public IExecutionLog
{
private:
	std::list<HostEvent> events = {};

public:
	void record(std::string eventKey, EventType eventType);

	double getEventDuration(std::string eventKey, DurationsMeasureUnit measureUnit);

	std::list<ReportEvent> getReport(DurationsMeasureUnit measureUnit);
};

