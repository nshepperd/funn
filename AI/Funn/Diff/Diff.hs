{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
module AI.Funn.Diff.Diff (
  Additive(..), (##), unit,
  Derivable(..),
  Diff(..),
  runDiff_, runDiffForward,
  runDiffD,
  first, second, (>>>),
  assocL, assocR, swap,
  fst, snd, dup
  ) where

import           Prelude hiding ((.), id, fst, snd)

import           Control.Applicative
import           Control.Category
import           Control.Monad
import           Data.Foldable
import           Data.Functor.Identity
import           Data.Vector (Vector)
import qualified Data.Vector.Generic as V

import AI.Funn.Space

class Derivable a where
  type family D a :: *

-- Diff represents the type of differentiable functions
newtype Diff m a b = Diff {
  runDiff :: a -> m (b, D b -> m (D a))
  }

instance Derivable Double where
  type D Double = Double

instance Derivable () where
  type D () = ()

instance Derivable Int where
  type D Int = ()

instance (Derivable a, Derivable b) => Derivable (a, b) where
  type D (a, b) = (D a, D b)

instance (Derivable a, Derivable b, Derivable c) => Derivable (a, b, c) where
  type D (a, b, c) = (D a, D b, D c)

instance Derivable a => Derivable (Vector a) where
  type D (Vector a) = Vector (D a)

type DiffV m a = (Derivable a, Additive m (D a))

runDiff_ :: Diff Identity a b -> a -> (b, D b -> D a)
runDiff_ (Diff f) a = let (b, dba) = runIdentity (f a) in
                       (b, runIdentity . dba)

runDiffForward :: Monad m => Diff m a b -> a -> m b
runDiffForward d a = do (b, _) <- runDiff d a
                        return b

runDiffD :: Monad m => Diff m a b -> a -> D b -> m (b, D a)
runDiffD diff a db = do (b, k) <- runDiff diff a
                        da <- k db
                        return (b, da)

diff_id :: (Applicative m) => Diff m a a
diff_id = Diff (\a -> pure (a, \db -> pure db))

diff_connect :: (Monad m) => Diff m a b -> Diff m b c -> Diff m a c
diff_connect mab mbc = Diff run
  where
    run a = do (b, dba) <- runDiff mab a
               (c, dcb) <- runDiff mbc b
               return (c, dcb >=> dba)

instance Monad m => Category (Diff m) where
  id = diff_id
  (.) = flip diff_connect

first :: (Monad m) => Diff m a b -> Diff m (a,c) (b,c)
first net = Diff run
  where
    run (a, c) = do (b, dba) <- runDiff net a
                    let back (db, dc) = do
                          da <- dba db
                          return (da, dc)
                    return ((b, c), back)

second :: (Monad m) => Diff m a b -> Diff m (c,a) (c,b)
second net = Diff run
  where
    run (c, a) = do (b, dba) <- runDiff net a
                    let back (dc, db) = do
                          da <- dba db
                          return (dc, da)
                    return ((c, b), back)

assocL :: (Applicative m) => Diff m (a,(b,c)) ((a,b),c)
assocL = Diff run
  where
    run (a,(b,c)) = let back ((da,db),dc) = pure (da,(db,dc)) in
                     pure (((a,b),c), back)

assocR :: (Applicative m) => Diff m ((a,b),c) (a,(b,c))
assocR = Diff run
  where
    run ((a,b),c) = let back (da,(db,dc)) = pure ((da,db),dc) in
                     pure ((a,(b,c)), back)

swap :: (Applicative m) => Diff m (a,b) (b,a)
swap = Diff run
  where
    run (a,b) = pure ((b,a), \(db,da) -> pure (da,db))

fst :: (Applicative m, Zero m (D b)) => Diff m (a,b) a
fst = Diff run
  where
    run (a,_) = let back da = fmap ((,) da) zero
                in pure (a, back)

snd :: (Applicative m, Zero m (D a)) => Diff m (a,b) b
snd = Diff run
  where
    run (_,b) = let back db = fmap (\da -> (da, db)) zero
                in pure (b, back)

dup :: (Applicative m, Semi m (D a)) => Diff m a (a,a)
dup = Diff run
  where
    run a = pure ((a,a), backward)
    backward (da1,da2) = plus da1 da2
