module AI.Funn.RNN (runRNN, runRNNIO, rnn) where
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ForeignFunctionInterface #-}

import           Control.Applicative
import           Control.Monad
import           Data.Foldable
import           Data.Traversable

import           Data.Coerce
import           Debug.Trace

import           Foreign.C
import           Foreign.Ptr
import           System.IO
import           System.IO.Unsafe

import           Data.Functor.Identity

import           Data.Vector (Vector)
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M
import qualified Numeric.LinearAlgebra.HMatrix as HM

import           AI.Funn.Network

foreign import ccall "vector_add" ffi_vector_add :: CInt -> Ptr Double -> Ptr Double -> IO ()

{-# NOINLINE vector_add #-}
vector_add :: M.IOVector Double -> S.Vector Double -> IO ()
vector_add tgt src = do M.unsafeWith tgt $ \tbuf -> do
                          S.unsafeWith src $ \sbuf -> do
                            ffi_vector_add (fromIntegral n) tbuf sbuf
  where
    n = V.length src

addToIO :: M.IOVector Double -> [Parameters] -> IO ()
addToIO target ys = go target (coerce ys :: [S.Vector Double])
  where
    go target [] = return ()
    go target (v:vs) = do
      vector_add target v
      go (M.drop (V.length v) target) vs

addTo :: Parameters -> [Parameters] -> Parameters
addTo (Parameters xs) ys = Parameters $ unsafePerformIO body
  where
    body = do target <- V.thaw xs
              addToIO target ys
              V.unsafeFreeze target

addParameters :: Parameters -> Parameters -> Parameters
addParameters (Parameters x) (Parameters y) = Parameters (x + y)

scaleParameters :: Double -> Parameters -> Parameters
scaleParameters x (Parameters y) = Parameters (HM.scale x y)

rnn :: (Monad m) => Network m (s,i) s -> [i] -> Network m s s
rnn layer inputs = Network ev (params layer) (initialise layer)
  where
    ev params s = do (new_s, k) <- go params s inputs
                     let backward ds_new = do
                           (ds, dpar) <- k ds_new
                           return (ds, [dpar])
                     return (new_s, 0, backward)

    go params s [] = return (s, \ds -> return (ds, Parameters (V.replicate p 0)))
    go params s (i:is) = do (s_1, _, k1) <- evaluate layer params (s,i)
                            (s_2, k) <- go params s_1 is
                            let backward ds_3 = do
                                  (ds_2, dp2) <- k ds_3
                                  ((ds_1, _), dp1) <- k1 ds_2
                                  return (ds_1, dp2 `addTo` dp1)
                            return (s_2, backward)

    n = length inputs
    p = params layer

runRNN :: (Monad m) => s -> Network m (s,i) s -> Parameters -> Network m (s,o) () -> Parameters -> [i] -> o -> m (Double, D s, D Parameters, D Parameters)
runRNN s_init layer p_layer final p_final inputs o = do (c, ds, d_layer, d_final) <- go s_init inputs
                                                        return $ (c, ds, scaleParameters (1 / fromIntegral n) d_layer, d_final)
  where
    go s [] = do ((), cost, k) <- evaluate final p_final (s, o)
                 ((ds, _), l_dp_final) <- k ()
                 return (cost, ds, Parameters (V.replicate (params layer) 0), fold l_dp_final)

    go s (i:is) = do (s_new, _, k) <- evaluate layer p_layer (s, i)
                     (cost, ds, dp_layer, dp_final) <- go s_new is
                     ((ds2, _), l_dp_layer2) <- k ds
                     return (cost, ds2, dp_layer `addTo` l_dp_layer2, dp_final)

    n = length inputs

-- add parameters derivative in a mutable vector to avoid copying
runRNNIO :: s -> Network Identity (s,i) s -> Parameters -> Network Identity (s,o) () -> Parameters -> [i] -> o -> IO (Double, D s, D Parameters, D Parameters)
runRNNIO s_init layer p_layer final p_final inputs o = do d_layer <- M.replicate (params layer) 0
                                                          (c, ds, d_final) <- go d_layer s_init inputs
                                                          d_layer' <- V.unsafeFreeze d_layer
                                                          return $ (c, ds, scaleParameters (1 / fromIntegral n) (Parameters d_layer'), d_final)
  where
    go _       s [] = do let Identity ((), cost, k) = evaluate final p_final (s, o)
                             Identity ((ds, _), l_dp_final) = k ()
                         return (cost, ds, fold l_dp_final)

    go d_layer s (i:is) = do let Identity (s_new, _, k) = evaluate layer p_layer (s, i)
                             (cost, ds, dp_final) <- go d_layer s_new is
                             let Identity((ds2, _), l_dp_layer) = k ds
                             addToIO d_layer l_dp_layer
                             return (cost, ds2, dp_final)

    n = length inputs
