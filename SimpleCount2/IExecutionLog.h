#pragma once
#include <string>
#include <list>
#include "ReportEvent.h"


class IExecutionLog
{
public:
	virtual void record(std::string eventKey, EventType eventType);
	
	virtual double getEventDuration(std::string eventKey, DurationsMeasureUnit measureUnit);

	virtual std::list<ReportEvent> getReport(DurationsMeasureUnit measureUnit);

	std::string str(DurationsMeasureUnit durationsMeasureUnit);

	std::string str() { return str(MILLISECONDS); }
};


