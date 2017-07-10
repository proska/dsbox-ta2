import json
import pprint
import pkgutil
import random

class Ontology(object):
    ONTOLOGY_FILE='ontology.json'

    def __init__(self):
        self.load()

    def load(self):
        text = pkgutil.get_data('dsbox.planner.levelone', self.ONTOLOGY_FILE)
        print(type(text))
        content = json.loads(text.decode())
        self.task = content['TaskOntology']
        self.learningType = content['LearningTypeOntology']
        self.algo = content['MachineLearningAlgorithmOntology']

class Primitives(object):
    PRIMITIVE_FILE='primitives.json'

    def __init__(self):
        self.load()

    def load(self):
        text = pkgutil.get_data('dsbox.planner.levelone', self.PRIMITIVE_FILE)
        content = json.loads(text.decode())
        self.primitives = content['Primitives']

    def filterEquality(self, aspect, name):
        result = [p for p in self.primitives if p[aspect]==name]
        return result

    def filterByTask(self, name):
        return self.filterEquality('Task', name)
    
    def filterByLearningType(self, name):
        return self.filterEquality('LearningType', name)
    
    def filterByAlgo(self, name):
        return self.filterEquality('MachineLearningAlgorithm', name)

    def getByName(self, name):
        for p in self.primitives:
            if p['Name'] == name:
                return p
        return None
    
ONTOLOGY = Ontology()
PRIMITIVES = Primitives()

class ConfigurationSpace(object):
    def __init__(self, space):
        self.space = space

    def getRandomConfiguration(self, seed=None):
        if seed:
            random.seed(seed)
        components = []
        for componentSpace in self.space:
            i = random.randrange(len(componentSpace))
            component = componentSpace[i]
            components.append(component)
        return components

        
class Pipeline(object):
    def __init__(self, componentList):
        self.componentList = componentList

    @classmethod
    def getRandomPipeline(cls, configurationSpace):
        return Pipeline(configurationSpace.getRandomConfiguration())

    def __str__(self):
        str = 'Pipeline(' + ','.join([c['Name'] for c in self.componentList]) + ')'
        return str
        
def generatePipelines(n=5, learningType='Classification', evaluatorName='F1', primitives=PRIMITIVES):

    preprocess = primitives.filterByTask('DataPreprocessing')
    feature = primitives.filterByTask('FeatureExtraction')
    learner = primitives.filterByLearningType(learningType)
    algos = [(l['MachineLearningAlgorithm'], l) for l in learner]
    evaluator = [ primitives.getByName(evaluatorName) ]

    # pprint.pprint(preprocess)
    # pprint.pprint(feature)
    # pprint.pprint(learner)
    # pprint.pprint(evaluator)

    configurationSpace = ConfigurationSpace((preprocess, feature, learner, evaluator))

    pipelines = []
    for i in range(n):
        pipeline = Pipeline.getRandomPipeline(configurationSpace)
        pipelines.append(pipeline)

    return pipelines

