import argparse
import sys
import os
import re

from record import Record

def getSecondArgumentOfPredicate(groundedFormula, predicateName):
    match = re.search('_obj\(([^,]+), ([^)]+)\)', groundedFormula)
    if match is not None:
        return match.group(2)
    else:
        return None

class AbstractRule:
    
    def isApplicable(self, record):
        return False
    
    def convert(self, record):
        return record
    
    def getDescription(self):
        return 'abstract rule, always not applicable'
    
    def __str__(self):
        return self.getDescription()

class AbstractOtherDetObjSubjRule(AbstractRule):
    
    def isApplicable(self, record):
        return (record.questionType == 'other'
                and record.formula == '_det(A, B);_obj(C, D);_subj(C, A)')
    
    def getDescription(self):
        return ('applicable to _det(A, B);_obj(C, D);_subj(C, A) questions; '
                'no action')

class YesNoPredAdjAB(AbstractRule):
    
    def isApplicable(self, record):
        return (record.questionType == 'yes/no' 
                and record.formula == '_predadj(A, B)')
    
    def getDescription(self):
        return '_predadj(A, B) yes/no questions returned as is'

class WhatColorIsYes(AbstractOtherDetObjSubjRule):
    
    def isApplicable(self, record):
        return (super().isApplicable(record) 
         and record.groundedFormula.endswith(';_subj(be, color)')
         and (not record.answer.startswith('no '))
         and len(record.answer.split()) == 1)
    
    def convert(self, record):
        object = getSecondArgumentOfPredicate(record.groundedFormula, '_obj')
        if object is None:
            return None
        color = record.answer
        
        newRecord = Record.fromOther(record)
        newRecord.question = 'Is the {} {}?'.format(object, color)
        newRecord.questionType = 'yes/no'
        newRecord.answer = 'yes'
        newRecord.formula = '_predadj(A, B)'
        newRecord.groundedFormula = '_predadj({}, {})'.format(object, color)
        return newRecord

class WhatColorIsNo(AbstractOtherDetObjSubjRule):
    
    def isApplicable(self, record):
        return (super().isApplicable(record) 
         and record.groundedFormula.endswith(';_subj(be, color)')
         and record.answer.startswith('no ')
         and len(record.answer.split()) == 2)
    
    def convert(self, record):
        object = getSecondArgumentOfPredicate(record.groundedFormula, '_obj')
        if object is None:
            return None
        color = record.answer
        
        newRecord = Record.fromOther(record)
        newRecord.question = 'Is the {} {}?'.format(object, color)
        newRecord.questionType = 'yes/no'
        newRecord.answer = 'no'
        newRecord.formula = '_predadj(A, B)'
        newRecord.groundedFormula = '_predadj({}, {})'.format(object, color)
        return newRecord

parser = argparse.ArgumentParser(description='Generate questions set for '
                                 'training')
parser.add_argument('--input', '-i', dest='inputFileName', action='store',
    type = str, help='input file name stdin if absent')
parser.add_argument('--output', '-o', dest='outputFileName', action='store',
    type = str, help='ouput file name stdout if absent')
args = parser.parse_args()

rules = [ YesNoPredAdjAB(), WhatColorIsYes(), WhatColorIsNo() ]

with (sys.stdin if args.inputFileName is None 
      else open(args.inputFileName, 'r')) as input:
    
    with (sys.stdout if args.outputFileName is None 
          else open(args.outputFileName, 'w')) as output:
        
        for line in input:
            record = Record.fromString(line.strip())
            
            for rule in rules:
                
                if rule.isApplicable(record):
                    newRecord = rule.convert(record)
                    if newRecord is None:
                        continue
                    output.write(newRecord.toString())
                    output.write(os.linesep)
