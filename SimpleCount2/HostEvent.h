#pragma once
#include "ReportEvent.h"

#include <chrono>
#include <ctime>

/// <summary>
/// Evento specifico de  
/// </summary>
class HostEvent : public IReportEvent {

private:
	IReportEvent reportEvent;
	std::chrono::time_point<std::chrono::system_clock> timePoint;

public:

	HostEvent(std::string eventKey, EventType eventType){

		reportEvent = ReportEvent(eventKey, eventType);
		timePoint = std::chrono::system_clock::now();
	}


	std::string getEventKey() { return reportEvent.getEventKey(); }
	EventType getEventType() { return reportEvent.getEventType(); };
	std::chrono::time_point<std::chrono::system_clock> getTimePoint() { return timePoint; }

	std::string getTimePointStr();
	std::string str();
};

double calculateDuration(HostEvent startEvent, HostEvent endEvent, DurationsMeasureUnit measureUnit);


