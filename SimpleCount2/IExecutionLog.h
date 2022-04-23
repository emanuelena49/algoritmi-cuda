#pragma once
#include <string>
#include <list>
#include "ReportEvent.h"


class IExecutionLog
{
public:
	virtual void record(std::string eventKey, EventType eventType) = 0;
	
	virtual double getEventDuration(std::string eventKey, DurationsMeasureUnit measureUnit) = 0;

	virtual std::list<ReportEvent> getReport(DurationsMeasureUnit measureUnit) = 0;

	std::string str(DurationsMeasureUnit durationsMeasureUnit);

	std::string str() { return str(MILLISECONDS); }
};


