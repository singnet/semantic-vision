import importlib.util
import os
import math
import zipfile
from opencog.scheme_wrapper import scheme_eval_as, scheme_eval


def currentDir(filePath):
    return str(os.path.dirname(os.path.realpath(filePath)))


def addLeadingZeros(number, requriedLength):
    result = ''
    nZeros = int((requriedLength - 1) - math.floor(math.log10(int(number))))
    for _ in range(0, nZeros):
        result += '0'
    return result + str(number)


def loadDataFromZipOrFolder(folderOrZip, fileName, loadProcedure):
    if (os.path.isdir(folderOrZip)):
        with open(folderOrZip + '/' + fileName, 'rb') as file:
            return loadProcedure(file)
    else:
        with zipfile.ZipFile(folderOrZip, 'r') as archive:
            with archive.open(fileName) as file:
                return loadProcedure(file)


def initialize_atomspace_by_facts(atomspaceFileName=None):
    atomspace = scheme_eval_as('(cog-atomspace)')
    scheme_eval(atomspace, '(use-modules (opencog))')
    scheme_eval(atomspace, '(use-modules (opencog exec))')
    scheme_eval(atomspace, '(use-modules (opencog query))')
    scheme_eval(atomspace, '(use-modules (opencog logger))')
    scheme_eval(atomspace, '(add-to-load-path ".")')
    if atomspaceFileName is not None:
        scheme_eval(atomspace, '(load-from-path "' + atomspaceFileName + '")')
    return atomspace

