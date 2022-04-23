#pragma once
#include <string>

enum EventType { START, END };

enum DurationsMeasureUnit { SECONDS, MILLISECONDS, MICROSECONDS };

std::string eventTypeStr(EventType et);

std::string durationMeasureUnitStr(DurationsMeasureUnit mu);

/// <summary>
/// Un generico evento del report. Contiene solo una chiave e un tipo.
/// Può essere convertito in stringa.
/// </summary>
class ReportEvent
{
private:
	std::string eventKey;
	EventType eventType;

public:
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
class ReportEndEvent : ReportEvent 
{
private:
	double eventDuration; 
	DurationsMeasureUnit durationMeasureUnit;

public:
	ReportEndEvent(std::string key, double intervalDuration, DurationsMeasureUnit durationUnit) :
		ReportEvent(key, END) {
		eventDuration = intervalDuration;
		durationMeasureUnit = durationUnit;
	}

	double getEventDuration() { return eventDuration; }
	DurationsMeasureUnit getDurationMeasureUnit() { return durationMeasureUnit; }

	std::string str();
};

