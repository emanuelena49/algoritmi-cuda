#pragma once
#include <string>
#include <list>

enum LogRowType {
	START_EVENT, STOP_EVENT, UNDEFINED_EVENT 
};

struct LogRow {
	std::string key;
	std::string note;
	LogRowType type;
	// todo: time
	// todo: difference (for stop)
};

using Log = std::list<LogRow>;

void addEventHost(Log log, std::string key, std::string note, LogRowType);

void addStartEventHost(Log log, std::string key, std::string note);

void addEndEventHost(Log log, std::string key, std::string note);