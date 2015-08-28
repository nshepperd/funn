{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ForeignFunctionInterface #-}
module AI.Funn.RNN (runRNN, rnn, rnnBig, zipWithNetwork_, rnnX) where

import           Control.Applicative
import           Control.Applicative.Backwards
import           Control.Monad
import           Control.Monad.State.Lazy
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

sumParameterList :: Foldable f => Int -> f [Parameters] -> Parameters
sumParameterList n xss = Parameters $ unsafePerformIO go
  where
    go = do target <- M.replicate n 0
            traverse_ (addToIO target) xss
            V.unsafeFreeze target

addParameters :: Parameters -> Parameters -> Parameters
addParameters (Parameters x) (Parameters y) = Parameters (x + y)

scaleParameters :: Double -> Parameters -> Parameters
scaleParameters x (Parameters y) = Parameters (HM.scale x y)

instance Derivable a => Derivable (Vector a) where
  type D (Vector a) = Vector (D a)

traverseBack :: (Traversable t, Applicative f) => (a -> f b) -> t a -> f (t b)
traverseBack f = forwards . traverse (Backwards . f)

rnn :: (Monad m) => Network m (s,i) s -> Network m (s, Vector i) s
rnn layer = Network ev (params layer) (initialise layer)
  where
    ev pars (s, inputs) = do (ks, s') <- runStateT (traverse (go_forward pars) inputs) s
                             let α = 1 / fromIntegral (1 + V.length inputs)
                                 backward ds' = do
                                   (back, ds) <- runStateT (traverseBack go_backward ks) ds'
                                   let dpar = sumParameterList p (V.map snd back)
                                       dis = V.map fst back
                                   return ((ds, dis), [scaleParameters α dpar])
                             return (s', 0, backward)

    go_forward pars i = do s <- get
                           (s', _, k) <- lift (evaluate layer pars (s,i))
                           put s'
                           return k

    go_backward k = do ds' <- get
                       ((ds, di), dparl) <- lift (k ds')
                       put ds
                       return (di, dparl)

    p = params layer

rnnBig :: (ds ~ D s, VectorSpace ds, Monad m) => Network m (s,i) s -> Network m (s, Vector i) (Vector s)
rnnBig layer = Network ev (params layer) (initialise layer)
  where
    ev pars (s, inputs) = do stuff <- evalStateT (traverse (go_forward pars) inputs) s
                             let out = V.cons s (V.map fst stuff)
                                 ks = V.map snd stuff
                                 α = 1 / fromIntegral (1 + V.length inputs)
                                 backward dss = do
                                   (back, ds) <- runStateT (traverseBack go_backward (V.zip ks (V.init dss))) (V.last dss)
                                   let dpar = sumParameterList p (V.map snd back)
                                       dis = V.map fst back
                                   return ((ds, dis), [scaleParameters α dpar])
                             return (out, 0, backward)

    go_forward pars i = do s <- get
                           (s', _, k) <- lift $ evaluate layer pars (s,i)
                           put s'
                           return (s', k)

    go_backward (k, dsA) = do ds' <- get
                              ((dsB, di), dparl) <- lift (k ds')
                              let ds = dsA ## dsB
                              put ds
                              return (di, dparl)

    p = params layer

rnnX :: (Monad m) => Network m (s,a) (s,b) -> Network m (s, Vector a) (s, Vector b)
rnnX layer = Network ev (params layer) (initialise layer)
  where
    ev pars (s, inputs) = do (stuff, s') <- runStateT (traverse (go_forward pars) inputs) s
                             let out = V.map fst stuff
                                 ks = V.map snd stuff
                                 α = 1 / fromIntegral (V.length inputs)
                                 backward (ds', dbs) = do
                                   (back, ds) <- runStateT (traverseBack go_backward (V.zip ks dbs)) ds'
                                   let dpar = if V.length inputs > 0 then
                                                scaleParameters α $ sumParameterList p (V.map snd back)
                                              else
                                                Parameters $ V.replicate p 0
                                       dis = V.map fst back
                                   return ((ds, dis), [dpar])
                             return ((s', out), 0, backward)

    go_forward pars a = do s <- get
                           ((s', b), _, k) <- lift $ evaluate layer pars (s,a)
                           put s'
                           return (b, k)

    go_backward (k, db) = do ds' <- get
                             ((ds, di), dparl) <- lift (k (ds', db))
                             put ds
                             return (di, dparl)

    p = params layer

runRNN :: (Monad m) => s -> Network m (s,i) s -> Parameters -> Network m (s,o) () -> Parameters -> [i] -> o -> m (Double, D s, D Parameters, D Parameters)
runRNN s_init layer p_layer final p_final inputs o = do
  (s, _, kl) <- evaluate (rnn layer) p_layer (s_init, V.fromList inputs)
  ((), cost, kf) <- evaluate final p_final (s,o)
  ((ds,_), d_final) <- kf ()
  ((ds_init,_), d_layer) <- kl ds
  return (cost, ds_init, fold d_layer, fold d_final)

zipWithNetwork_ :: (Monad m) => Network m (a, b) () -> Network m (Vector a, Vector b) ()
zipWithNetwork_ network = Network ev (params network) (initialise network)
  where
    ev pars (as, bs) = do stuff <- traverse (go_forward pars) (V.zip as bs)
                          let cost = V.sum (V.map fst stuff)
                              ks = V.map snd stuff
                              α = 1 / fromIntegral (1 + V.length as)
                              backward () = do
                                back <- traverse go_backward ks
                                let das = V.map (fst.fst) back
                                    dbs = V.map (snd.fst) back
                                    dpar = sumParameterList (params network) (V.map snd back)
                                return ((das, dbs), [scaleParameters α dpar])
                          return ((), cost, backward)

    go_forward pars (a,b) = do (_, cost, k) <- evaluate network pars (a,b)
                               return (cost, k)

    go_backward k = k ()
