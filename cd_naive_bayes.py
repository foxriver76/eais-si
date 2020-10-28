from sklearn.base import BaseEstimator, ClassifierMixin
from skmultiflow.bayes.naive_bayes import NaiveBayes
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import  EDDM
from skmultiflow.drift_detection.ddm import DDM
from inc_pca import IncPCA
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
class IncPCNB(ClassifierMixin, BaseEstimator):
    def __init__(self,n_components,subdim=2,threshold=0.6,forgetting_factor=1):
        self.classifier = NaiveBayes()
        self.n_components = n_components
        self.init_drift_detection = True
        self.threshold = threshold
        self.n_detections = 0
        self.subdim = subdim
        self.pca = IncPCA(n_components=self.n_components, forgetting_factor=forgetting_factor)
        self.eigenkontext = None

    def partial_fit(self, X, y, classes=None):
            """
            Calls the MultinomialNB partial_fit from sklearn.
            ----------
            x : array-like, shape = [n_samples, n_features]
              Training vector, where n_samples in the number of samples and
              n_features is the number of features.
            y : array, shape = [n_samples]
              Target values (integers in classification, real numbers in
              regression)
            Returns
            --------
            """
            # if self.concept_drift_detection(X):
            #     self.classifier.reset()
            # Remarks:
            # Partial Fit always creates new pcs with partial fit and last pcs from
            # last iteration are compared
            self.init_eigenkontext(X)
            print(self.eigenkontext.shape)
            recent_pcs  = self.compute_recent_eigenkontext(X)
            cossim = self.ordered_cosine_similarity(recent_pcs,self.pca.get_loadings())
            self.eigenkontext = self.determine_new_eigencontext(cossim)
            print(self.eigenkontext.shape)
            X_reduced = self.reduce_feature_space(X)
            print(self.eigenkontext.shape)
            self.pca.partial_fit(X)
            self.classifier.partial_fit(X_reduced, y,classes)
            print(self.eigenkontext.shape)
            return self

    def reduce_feature_space(self,X):
        return X @ self.eigenkontext.T

    def init_eigenkontext(self,X):
        if self.eigenkontext is None:
            self.pca.partial_fit(X)
            self.eigenkontext = self.pca.get_loadings()[:self.subdim,:]

    def determine_new_eigencontext(self,cossim):
        for threshold in np.arange(1.0,0.6,-0.01):
            idx = np.where(cossim > threshold)[0]
            if self.subdim == len(idx):
                return self.pca.get_loadings()[idx]

    def predict(self, X):
        X = self.reduce_feature_space(X)
        return self.classifier.predict(X)

    def concept_drift_detection(self, X):
        self.drift_detected = True
        return self.drift_detected

    def compute_recent_eigenkontext(self, X):
        pca = IncPCA(n_components=self.n_components, forgetting_factor=0)
        pca.partial_fit(X)
        return pca.get_loadings()

    def ordered_cosine_similarity(self,pc1,pc2):
        return np.diag(cosine_similarity(pc1,pc2))

if __name__ == "__main__":

    from skmultiflow.data import MIXEDGenerator
    from skmultiflow.data import ConceptDriftStream
    from skmultiflow.meta import OzaBaggingAdwin
    from skmultiflow.lazy import KNN
    from skmultiflow.evaluation import EvaluatePrequential
    s1 = MIXEDGenerator(classification_function = 1, random_state= 112, balance_classes = False)
    s2 = MIXEDGenerator(classification_function = 0, random_state= 112, balance_classes = False)


    stream = ConceptDriftStream(stream=s1, drift_stream=s2,random_state=None,alpha=90.0, position=2000,width=1)


    cls = IncPCNB(stream.n_features)


    evaluator = EvaluatePrequential(show_plot=True,max_samples=10000,
    restart_stream=True,batch_size=10,metrics=[ 'accuracy', 'kappa', 'kappa_m'])

    evaluator.evaluate(stream=stream, model=[cls],model_names=["test"])