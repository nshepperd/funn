{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.Models.RNNChar (Model, ModelPars, ParsWrapped, buildModel, initialize, readPars, storePars) where

import qualified Codec.CBOR.Decoding as C
import qualified Codec.CBOR.Read as C
import qualified Codec.CBOR.Write as C
import qualified Codec.Serialise.Class as C
import           Control.Monad.IO.Class
import qualified Data.ByteString.Lazy as BL
import           Data.Constraint
import           Data.Proxy
import           Data.Random
import           Data.Type.Equality
import           Data.Vector (Vector)
import qualified Data.Vector.Generic as V
import           GHC.TypeLits
import           Unsafe.Coerce

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

type family Pars (size :: Nat) :: Nat

parDict :: KnownNat size => Proxy size -> Dict (KnownNat (Pars size))
parDict size = case stepNetwork @IO size of
                 Network (Proxy :: Proxy ps) _ _ -> unsafeCoerce (Dict :: Dict (KnownNat ps))

data Model m size where
  Model :: KnownNat (Pars size) => {
    modelStep :: Diff m (Blob (Pars size), (StepState size, Blob 256)) (StepState size, Blob 256),
    modelRun :: Diff m ((Blob (Pars size), StepState size), Vector (Blob 256)) (Vector (Blob 256)),
    modelEval :: Diff m ((Blob (Pars size), StepState size), (Vector (Blob 256), Vector Int)) Double
    } -> Model m size

buildModel :: (Monad m, KnownNat size) => Proxy size -> Model m size
buildModel size = case parDict size of
                    Dict ->
                      let Just (diff_step, _) = openNetwork (stepNetwork size)
                          Just (diff_run, _) = openNetwork (allNetwork size)
                          Just (diff_eval, _) = openNetwork (evalNetwork size)
                      in (Model diff_step (Diff.assocR >>> diff_run)
                           (Diff.assocR >>> diff_eval))

data ModelPars size where
  ModelPars :: KnownNat (Pars size) => Blob (Pars size) -> StepState size -> ModelPars size

sampleIO :: MonadIO m => RVar a -> m a
sampleIO v = liftIO (runRVar v StdRandom)

initialize :: KnownNat size => Proxy size -> IO (ModelPars size)
initialize size = case parDict size of
                    Dict -> case openNetwork (stepNetwork @IO size) of
                      Just (_, initrvar) -> do
                        p <- sampleIO initrvar
                        s0 <- Blob.generate (pure 0)
                        s1 <- Blob.generate (pure 0)
                        return (ModelPars p (s0, s1))

data ParsWrapped = forall size. KnownNat size => ParsWrapped (ModelPars size)

instance C.Serialise ParsWrapped where
  encode (ParsWrapped (ModelPars p s0 :: ModelPars size)) =
    C.encode (natVal (Proxy @ size), p, s0)
  decode = do
    3 <- C.decodeListLen
    size <- C.decodeInteger
    withNat size $ \(proxy :: Proxy size) -> do
      case parDict proxy of
        Dict -> do
          p <- C.decode
          s0 <- C.decode
          return (ParsWrapped (ModelPars p s0 :: ModelPars size))

storePars :: forall size. KnownNat size => FilePath -> ModelPars size -> IO ()
storePars fname m = BL.writeFile fname (encodeToByteString (ParsWrapped m))

readPars :: FilePath -> IO ParsWrapped
readPars fname = decodeOrError  <$> BL.readFile fname
