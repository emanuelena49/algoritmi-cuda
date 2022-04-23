#include "HostExecutionLog.h"


void HostExecutionLog::record(std::string eventKey, EventType eventType) {

	HostEvent event = HostEvent(eventKey, eventType);
	events.push_back(event);
}

double HostExecutionLog::getEventDuration(std::string eventKey, DurationsMeasureUnit measureUnit) {

	// recupero record di inizio e di fine
	std::list<HostEvent>::iterator startIt, endIt, it;
	startIt = endIt = events.end();
	for (std::list<HostEvent>::iterator it = events.begin(); it != events.end(); ++it)
	{
		if ((*it).getEventKey() == eventKey && (*it).getEventType() == START) {
			startIt = it;
		}
		else if ((*it).getEventKey() == eventKey && (*it).getEventType() == END) {
			endIt = it;
		}
	}

	// se non esiste un record di inizio e/o un record di fine, non posso calcolare la durata
	if (startIt == events.end()) {
		return -1;
	}

	if (endIt == events.end()) {
		return -1;
	}

	return calculateDuration(*startIt, *endIt, measureUnit);
}


std::list<ReportEvent> HostExecutionLog::getReport(DurationsMeasureUnit measureUnit) {

	std::list<ReportEvent> report = {};

	// lista temporanea con gli eventi di start che uso per calcolare facilmente le durate
	std::list<HostEvent*> pendingStarts = {};

	for each (HostEvent e in events)
	{
		if (e.getEventType() == START) {

			// inserisco gli eventi di start in una lista temporanea che
			// mi facilita il recupero per il calcolo delle durate
			pendingStarts.push_front(&e);
			report.push_back(e);
		}
		else if (e.getEventType() == END) {

			// cerco un'eventuale evento di inizio salvato nella lista temporanea
			std::list<HostEvent*>::iterator eventualStartIt = pendingStarts.end();

			for (std::list<HostEvent*>::iterator it = pendingStarts.begin(); it != pendingStarts.end(); ++it)
			{
				if ((**it).getEventKey() == e.getEventKey()) {
					eventualStartIt = it;
				}
			}

			if (eventualStartIt != pendingStarts.end()) {

				// se lo trovo, decoro il record con la durata dell'evento...
				const HostEvent startEvent = **eventualStartIt;

				double duration = calculateDuration(startEvent, e, measureUnit);
				report.push_back(DurationDecoratorRecord(e, duration, measureUnit));
				// ... e rimuovo l'evento di inizio dalla lista temporanea

				pendingStarts.remove(*eventualStartIt);
			}
			else {

				// altrimenti, se non ho trovato il record di inizio, registro quello di 
				// fine normalmente (ignorando la durata)
				report.push_back(e);
			}
		}
	}

	return report;
	
	return {};
}


