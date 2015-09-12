module AI.Funn.Search where

import           Data.Sequence (Seq)
import qualified Data.Sequence as Seq

import           Data.List
import           Data.Ord

import           Debug.Trace

data CheckResult c = Equal c | Overflow | Wait

beamSearch :: (Eq c) => Int -> s -> (s -> [(c, s, Double)]) -> [c]
beamSearch n start step = go [(Seq.empty, start, 0)]
  where
    forward (cs, s, p) = [(cs Seq.|> c, next, p + q) | (c, next, q) <- step s]

    check states = let cs = map getHead states
                       len = case head states of (cs, _, _) -> Seq.length cs
                   in case head cs of
                       Just c
                         | all (== (Just c)) cs -> Equal c
                         | len > 50             -> Overflow
                         | otherwise            -> Wait
                       Nothing -> Wait

    go states = case check states of
                 Equal c -> c : go (map (\(cs, s, p) -> (Seq.drop 1 cs, s, p)) states)
                 Overflow -> go ([head states])
                 Wait -> let allnext = sortBy (comparing (Down . \(_,_,p) -> p)) (foldMap forward states)
                         in go (take n allnext)


    getHead (cs, _, _) = case Seq.viewl cs of
                          Seq.EmptyL -> Nothing
                          c Seq.:< _ -> Just c
