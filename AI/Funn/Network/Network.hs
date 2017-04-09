{-# LANGUAGE TypeFamilies, KindSignatures, FlexibleContexts #-}
{-# LANGUAGE BangPatterns, MultiParamTypeClasses, FlexibleInstances #-}
{-# LANGUAGE TypeOperators, GADTs, ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications, DataKinds #-}
module AI.Funn.Network.Network (
  Network(..),
  -- runNetwork, runNetwork', runNetwork_,
  liftDiff,
  first, second, (>>>),
  (***),
  assocL, assocR, swap
  ) where

import           Prelude hiding ((.), id)

import           Control.Applicative
import           Control.Monad
import           Control.Category
import           Data.Foldable
import           Data.Monoid
import           Data.Proxy

import           Control.DeepSeq
import qualified Data.Binary as B
import           Data.Functor.Identity
import           Data.Random
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import           GHC.TypeLits

import           AI.Funn.Common
import           AI.Funn.Diff.Diff (Additive(..), Derivable(..), Diff(..))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Flat.Blob (Blob(..))
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Flat.Flat
import           AI.Funn.SomeNat

data Network m a b where
  Network :: KnownNat p => Proxy p -> Diff m (Blob p, a) b -> RVar (Blob p) -> Network m a b

params :: Network m a b -> Int
params (Network p _ _) = fromIntegral (natVal p)

concatInit :: (KnownNat a, KnownNat b)
           => RVar (Blob a) -> RVar (Blob b)
           -> RVar (Blob (a + b))
concatInit = liftA2 Blob.cat

first :: (Monad m) => Network m a b -> Network m (a,c) (b,c)
first (Network p diff init) = Network p run init
  where
    run = Diff.assocL >>> Diff.first diff

second :: (Monad m) => Network m a b -> Network m (c,a) (c,b)
second (Network p diff init) = Network p run init
  where
    run = Diff.second Diff.swap >>> Diff.assocL >>> Diff.first diff >>> Diff.swap

liftDiff :: (Monad m) => Diff m a b -> Network m a b
liftDiff diff = Network (Proxy @ 0) run (pure (Blob.fromList []))
  where
    run = Diff (\(e, a) -> pure (a, \da -> pure (e, da))) >>> diff

assocL :: (Monad m) => Network m (a,(b,c)) ((a,b),c)
assocL = liftDiff Diff.assocL

assocR :: (Monad m) => Network m ((a,b),c) (a,(b,c))
assocR = liftDiff Diff.assocR

swap :: (Monad m) => Network m (a,b) (b,a)
swap = liftDiff Diff.swap

connect :: (Monad m) => Network m a b -> Network m b c -> Network m a c
connect (Network (p1 :: Proxy p1) diff1 init1) (Network (p2 :: Proxy p2) diff2 init2) =
  Network p diff init
  where
    p = Proxy @ (p1 + p2)
    init = concatInit init1 init2
    diff = Diff $ \(params, a) -> do
      let (par1, par2) = Blob.split params
      (b, k1) <- runDiff diff1 (par1, a)
      (c, k2) <- runDiff diff2 (par2, b)
      let backward dc = do
            (dpar2, db) <- k2 dc
            (dpar1, da) <- k1 db
            return (Blob.cat dpar1 dpar2, da)
      return (c, backward)

net_empty :: (Monad m) => Network m a a
net_empty = liftDiff id

-- idWith :: (Monad m) => proxy a -> Network m a a
-- idWith _ = net_empty

instance Monad m => Category (Network m) where
  id = net_empty
  (.) = flip connect

infixr 3 ***
one *** two = first one >>> second two
