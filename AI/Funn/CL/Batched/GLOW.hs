{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.CL.Batched.GLOW (
  Invertible(..),
  invert
  ) where


import           Control.Applicative
import           Control.Category
import           Control.Exception
import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.Monoid
import           Data.Proxy
import           Data.Random
import           Data.Traversable
import           GHC.TypeLits
import           Prelude hiding (id)
import           System.IO.Unsafe

import           AI.Funn.CL.Batched.BTensor (BTensor(..))
import qualified AI.Funn.CL.Batched.BTensor as BT
import           AI.Funn.CL.Batched.Param (Param(..))
import qualified AI.Funn.CL.Batched.Param as Param
import           AI.Funn.CL.Batched.Network (Network(..))
import qualified AI.Funn.CL.Batched.Network as Network
import           AI.Funn.CL.Tensor (Tensor)
import qualified AI.Funn.CL.Tensor as T
import qualified AI.Funn.CL.TensorLazy as TL
import           AI.Funn.Diff.Diff (Diff(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Indexed.Indexed
import           AI.Funn.Space

data Invertible m (ω :: Nat) (p :: Nat) a b = Invertible {
  invForward :: Network m ω p a b,
  invBackward :: Network m ω p b a
  }

params :: Invertible m ω p a b -> Proxy p
params _ = Proxy

batchSize :: Invertible m ω p a b -> Proxy ω
batchSize _ = Proxy

invert :: Invertible m ω p a b -> Invertible m ω p b a
invert (Invertible ab ba) = Invertible ba ab

-- Note: connect has asymmetry in the ordering of parameters.
--     one ~>> two  !=  invert (invert two ~>> invert one)
-- They compute the same mapping, but the former takes 'one's
-- parameters first, followed by 'two's parameters. The latter takes
-- the parameters in the opposite order.
-- This is so that 'invert one' takes the same parameters as 'one'.
connect :: (KnownDimsF [i,j,ω], Monad m) => Invertible m ω i a b -> Invertible m ω j b c -> Invertible m ω (i+j) a c
connect (Invertible ab ba) (Invertible bc cb) = Invertible ac ca
  where
    ac = ab ~>> bc
    ca = Network (Diff runBack) (netInit ac)
    runBack (par, c) = do
      let (p1, p2) = Param.split par
      (b, k2) <- runDiff (netDiff cb) (p2, c)
      (a, k1) <- runDiff (netDiff ba) (p1, b)
      let back da = do
            (dp1, db) <- k1 da
            (dp2, dc) <- k2 db
            return (Param.appendD dp1 dp2, dc)
      return (a, back)

instance (KnownNat ω, Monad m) => Indexed (Invertible m ω) where
  iid = Invertible iid iid
  (~>>) = connect

pfirst :: (KnownNat ω, Monad m) => Invertible m ω p a b -> Invertible m ω p (a,c) (b,c)
pfirst (Invertible ab ba) = Invertible (first ab) (first ba)

psecond :: (KnownNat ω, Monad m) => Invertible m ω p a b -> Invertible m ω p (c,a) (c,b)
psecond (Invertible ab ba) = Invertible (second ab) (second ba)

pswap :: (KnownNat ω, Monad m) => Invertible m ω 0 (a,b) (b,a)
pswap = Invertible swap swap

passocL :: (KnownNat ω, Monad m) => Invertible m ω 0 (a,(b,c)) ((a,b),c)
passocL = Invertible assocL assocR

passocR :: (KnownNat ω, Monad m) => Invertible m ω 0 ((a,b),c) (a,(b,c))
passocR = Invertible assocR assocL

instance (KnownNat ω, Monad m) => Assoc (Invertible m ω) where
  first = pfirst
  second = psecond
  swap = pswap
  assocL = passocL
  assocR = passocR
