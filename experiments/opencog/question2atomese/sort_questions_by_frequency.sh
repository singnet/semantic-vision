#!/bin/sh

##
## Usage: sort_questions_by_frequency.sh <parsed questions file>
## <parsed questions file> - result of questions2atomese.sh script
##

if test $# -lt 1 ; then
	egrep '^##' $0
	exit 1
fi

cat $1 | awk -F "::" '{print $2, $6}' | sort | uniq -c | sort -nr
