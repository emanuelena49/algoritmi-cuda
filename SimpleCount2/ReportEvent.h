#pragma once
#include <string>

enum EventType { START, END };

enum DurationsMeasureUnit { SECONDS, MILLISECONDS, MICROSECONDS };

std::string eventTypeStr(EventType et);

std::string durationMeasureUnitStr(DurationsMeasureUnit mu);

/// <summary>
/// Evento del report. Contiene solo una chiave e un tipo.
/// Può essere convertito in stringa.
/// </summary>
class ReportEvent
{
private:
	std::string eventKey;
	EventType eventType;

public:

	ReportEvent() { }

	ReportEvent(std::string key, EventType type) {
		eventKey = key; 
		eventType = type; 
	}

	std::string getEventKey() { return eventKey; }
	EventType getEventType() { return eventType; }

	std::string str();
};

/// <summary>
/// Un evento di fine (con eventType==END). 
/// Ai dati dell'evento normale aggiunge anche la durata complessiva
/// dell'evento, espressa in una certa unità di misura.
/// Anch'esso può essere convertito in stringa.
/// </summary>
class DurationDecoratorRecord : public ReportEvent
{
private:
	ReportEvent reportEvent;
	double eventDuration; 
	DurationsMeasureUnit durationMeasureUnit;

public:

	DurationDecoratorRecord(ReportEvent reportEvent, double intervalDuration, DurationsMeasureUnit durationUnit) : ReportEvent() {
		this->reportEvent = reportEvent;
		eventDuration = intervalDuration;
		durationMeasureUnit = durationUnit;
	}

	double getEventDuration() { return eventDuration; }
	DurationsMeasureUnit getDurationMeasureUnit() { return durationMeasureUnit; }

	std::string getEventKey() { return reportEvent.getEventKey(); }

	EventType getEventType() { return reportEvent.getEventType(); }

	std::string str();
};

