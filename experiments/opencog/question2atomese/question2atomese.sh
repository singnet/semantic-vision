#!/bin/sh
java -Dlogback.configurationFile=jar:file:./target/question2atomese-1.0-SNAPSHOT.jar!/logback.xml -jar ./target/question2atomese-1.0-SNAPSHOT.jar $*
