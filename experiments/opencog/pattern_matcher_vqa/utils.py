import importlib.util
import os
import math
import zipfile

def currentDir(filePath):
    return str(os.path.dirname(os.path.realpath(filePath)))

def importModuleFromFile(moduleName, fileName):
    moduleSpec = importlib.util.spec_from_file_location(moduleName, fileName)
    module = importlib.util.module_from_spec(moduleSpec)
    moduleSpec.loader.exec_module(module)
    return module

def addLeadingZeros(number, requriedLength):
    result = ''
    nZeros = int((requriedLength - 1) - math.floor(math.log10(int(number))))
    for _ in range(0, nZeros):
        result += '0'
    return result + str(number)

def loadDataFromZipOrFolder(folderOrZip, fileName, loadProcedure):
    if (os.path.isdir(folderOrZip)):
        with open(folderOrZip + '/' + fileName) as file:
            return loadProcedure(file)
    else:
        with zipfile.ZipFile(folderOrZip, 'r') as archive:
            with archive.open(fileName) as file:
                return loadProcedure(file)