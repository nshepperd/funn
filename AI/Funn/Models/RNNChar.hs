{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.Models.RNNChar (Model, ModelPars, buildModel, initialize) where

import           Control.Monad.IO.Class
import qualified Data.Binary as BL
import qualified Data.Binary.Put as BL
import qualified Data.Binary.Get as BL
import qualified Data.ByteString.Lazy as BL
import           Data.Proxy
import           Data.Random
import           Data.Type.Equality
import           Data.Vector (Vector)
import qualified Data.Vector.Generic as V
import           GHC.TypeLits

import           AI.Funn.Common
import           AI.Funn.Diff.Diff (Diff(..), Derivable(..), (>>>))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Flat.Blob (Blob, blob, getBlob)
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Flat.ParBox
import           AI.Funn.Network.Flat
import           AI.Funn.Network.LSTM
import           AI.Funn.Network.Mixing
import           AI.Funn.Network.Network
import qualified AI.Funn.Network.Network as Network
import           AI.Funn.Network.RNN
import           AI.Funn.SGD
import           AI.Funn.Space
import           AI.Funn.TypeLits

type StepState size = (Blob size, Blob size)

stepNetwork :: (Monad m, KnownNat size) => Proxy size -> Network m (StepState size, Blob 256) (StepState size, Blob 256)
stepNetwork Proxy = assocR
                    >>> second (mergeLayer >>> amixLayer (Proxy @ 5) >>> tanhLayer) -- (Blob n, Blob 4n)
                    >>> lstmLayer                                                   -- (Blob n, Blob n)
                    >>> second dupLayer >>> assocL >>> second (amixLayer (Proxy @ 5))

allNetwork :: (Monad m, KnownNat size) => Proxy size -> Network m (StepState size, Vector (Blob 256)) (Vector (Blob 256))
allNetwork size = scanlLayer (stepNetwork size) >>> sndNetwork

evalNetwork :: (Monad m, KnownNat size) => Proxy size -> Network m (StepState size, (Vector (Blob 256), Vector Int)) Double
evalNetwork size = assocL >>> first (allNetwork size) >>> zipLayer >>> mapLayer softmaxCost >>> vsumLayer

openNetwork :: forall m n a b. KnownNat n => Network m a b -> Maybe (Diff m (Blob n, a) b, RVar (Blob n))
openNetwork (Network p diff initial) =
  case sameNat p (Proxy @ n) of
    Just Refl -> Just (diff, initial)
    Nothing -> Nothing

data Model m size where
  Model :: KnownNat p => {
    modelStep :: Diff m (Blob p, (StepState size, Blob 256)) (StepState size, Blob 256),
    modelRun :: Diff m ((Blob p, StepState size), Vector (Blob 256)) (Vector (Blob 256)),
    modelEval :: Diff m ((Blob p, StepState size), (Vector (Blob 256), Vector Int)) Double
    } -> Model m size

buildModel :: (Monad m, KnownNat size) => Proxy size -> Model m size
buildModel size = case stepNetwork size of
                    Network _ diff_step _ ->
                      let Just (diff_run, _) = openNetwork (allNetwork size)
                          Just (diff_eval, _) = openNetwork (evalNetwork size)
                      in Model diff_step (Diff.assocR >>> diff_run) (Diff.assocR >>> diff_eval)

data ModelPars size where
  ModelPars :: KnownNat p => Blob p -> StepState size -> ModelPars size

sampleIO :: MonadIO m => RVar a -> m a
sampleIO v = liftIO (runRVar v StdRandom)

initialize :: KnownNat size => Proxy size -> IO (ModelPars size)
initialize size = case stepNetwork @IO size of
                    Network _ _ initrvar -> do
                      p <- sampleIO initrvar
                      s0 <- Blob.generate (pure 0)
                      s1 <- Blob.generate (pure 0)
                      return (ModelPars p (s0, s1))

storePars :: forall size. KnownNat size => FilePath -> ModelPars size -> IO ()
storePars fname (ModelPars pars (s0, s1)) = BL.writeFile fname bs
  where
    bput = do BL.putInt64le size
              BL.putInt64le p
              BL.put pars
              BL.put s0
              BL.put s1
    bs = BL.runPut bput
    size = fromIntegral (natVal (Proxy @ size))
    p = fromIntegral (natVal pars)

data ModelWrapped = forall size. KnownNat size => ModelWrapped (ModelPars size)

readPars :: FilePath -> IO ModelWrapped
readPars fname = BL.runGet bget <$> BL.readFile fname
  where
    bget = do size <- BL.getInt64le
              p <- BL.getInt64le
              withNat (fromIntegral size) $ \(Proxy :: Proxy size) -> do
                withNat (fromIntegral p) $ \(Proxy :: Proxy p) -> do
                  (pars :: Blob p) <- BL.get
                  (s0 :: Blob size) <- BL.get
                  (s1 :: Blob size) <- BL.get
                  return (ModelWrapped (ModelPars pars (s0,s1)))
