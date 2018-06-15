#!/bin/sh

# if RELEX env variable is defined then use its value; otherwise define it.
if test -z "$RELEX"; then
	RELEX=$OPENCOG/relex
fi
#echo "RELEX = $RELEX"

java \
	-Dwordnet.configfile=$RELEX/data/wordnet/file_properties.xml \
	-Drelex.tagalgpath=$RELEX/data/relex-tagging.algs \
	-Drelex.semalgpath=$RELEX/data/relex-semantic.algs \
	-Drelex.pennalgpath=$RELEX/data/relex-penn-tagging.algs \
	-Drelex.sfalgpath=$RELEX/data/relex-stanford.algs \
	-jar ./target/question2atomeese-1.0-SNAPSHOT.jar \
	$* 2>/dev/null
