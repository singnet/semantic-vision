Add linkgrammar.jar and relex.jar into local maven repository:
```
$ mvn install:install-file -Dfile=<linkgrammar-install-folder/linkgrammar.jar> -DgroupId=org.opencog -DartifactId=linkgrammar -Dversion=<linkgrammar.version> -Dpackaging=jar
$ mvn install:install-file -Dfile=<relex-install-folder/relex.jar> -DgroupId=org.opencog -DartifactId=relex -Dversion=<relex.version> -Dpackaging=jar
```

Run question2atomeese app:
```
$ RELEX=<path-to-relex-src-dir> ./question2atomeese.sh <path-to-parsed-questions>
```
