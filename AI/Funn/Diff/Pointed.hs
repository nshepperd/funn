{-# LANGUAGE TypeFamilies, KindSignatures, FlexibleContexts #-}
{-# LANGUAGE BangPatterns, GADTs, RankNTypes, ScopedTypeVariables #-}
{-# LANGUAGE MultiParamTypeClasses, FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances, DataKinds #-}
{-# LANGUAGE UndecidableInstances #-}
module AI.Funn.Diff.Pointed (runPointed, feed, Ref, joinP, Unpack(..), Pack(..), Var(..), (<--), refType) where

import           Prelude hiding ((.), id)

import           Control.Applicative
import           Control.Applicative.Backwards
import           Control.Monad
import           Control.Category
import           Data.Foldable
import           Data.Function hiding ((.), id)
import           Data.Monoid

import           Data.Maybe
import           Data.Proxy

import           Control.Lens
import           Data.Functor.Identity

import           Data.Random
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S

import           Control.Monad.State.Lazy

import           Data.Map (Map)
import qualified Data.Map.Strict as Map

import           Control.DeepSeq

import           Unsafe.Coerce

import           AI.Funn.Diff.Diff (Additive(..), Derivable(..), Diff(..))
import qualified AI.Funn.Diff.Diff as Diff

newtype Ref s a = Ref { getRef :: Int }
newtype Var s a = Var (Ref s a)

data Source s m where
  Single :: (Additive m (D a)) => Ref s a -> Ref s b -> Diff m a b -> Source s m
  Join :: (Additive m (D a), Additive m (D b)) => Ref s a -> Ref s b -> Ref s (a,b) -> Source s m
  Split :: (Additive m (D a), Additive m (D b)) => Ref s (a,b) -> Ref s a -> Ref s b -> Source s m

data Dynamic where
  Dynamic :: a -> Dynamic

data PData s m = PData { _pNextRef :: Int,
                         _pTable :: [Source s m] }
type Pointed s (m :: * -> *) a = State (PData s m) a

data Shape = U | B Shape Shape deriving (Show, Eq)

-- type family UnpackF (x :: *) :: Shape where
--   UnpackF (Var s a) = U
--   UnpackF (a, b) = B (UnpackF a) (UnpackF b)

class Unpack m s (shape :: Shape) i o | i -> s, o -> shape s, shape i -> o where
  unpack :: i -> Pointed s m o

instance Monad m => Unpack m s U (Ref s a) (Var s a) where
  unpack ref = return (Var ref)

instance (Monad m,
          Additive m (D a),
          Additive m (D b),
          Unpack m s x (Ref s a) a',
          Unpack m s y (Ref s b) b') =>
         Unpack m s (B x y) (Ref s (a, b)) (a', b') where
  unpack ref = do (a,b) <- splitP ref
                  a' <- unpack a
                  b' <- unpack b
                  return (a', b')

class Pack m s (shape :: Shape) i o | i -> s, o -> shape s, shape i -> o where
  pack :: o -> Pointed s m i

instance Monad m => Pack m s U (Ref s a) (Ref s a) where
  pack ref = return ref

instance (Monad m,
          Additive m (D a),
          Additive m (D b),
          Pack m s x (Ref s a) a',
          Pack m s y (Ref s b) b') =>
         Pack m s (B x y) (Ref s (a, b)) (a', b') where
  pack (a',b') = do a <- pack a'
                    b <- pack b'
                    joinP a b

pNextRef :: Lens' (PData s m) Int
pNextRef f (PData a b) = (`PData` b) <$> f a

pTable :: Lens' (PData s m) [Source s m]
pTable f (PData a b) = PData a <$> f b

newRef :: Pointed s m (Ref s a)
newRef = do next <- use pNextRef
            pNextRef .= next+1
            return (Ref next)

(<--) :: (Additive m (D a),
          Pack m s a_shape (Ref s a) a',
          Unpack m s b_shape (Ref s b) b') =>
         Diff m a b -> a' -> Pointed s m b'
(<--) network ref = do a <- pack ref
                       b <- feed a network
                       unpack b

refType :: forall a s m. Ref s a -> Pointed s m ()
refType _ = return ()

feed :: (Additive m (D a)) => Ref s a -> Diff m a b -> Pointed s m (Ref s b)
feed i network = do
  j <- newRef
  pTable %= (Single i j network :)
  return j

joinP :: (Additive m (D a), Additive m (D b)) => Ref s a -> Ref s b -> Pointed s m (Ref s (a,b))
joinP i j = do
  k <- newRef
  pTable %= (Join i j k :)
  return k

splitP :: (Additive m (D a), Additive m (D b)) => Ref s (a,b) -> Pointed s m (Ref s a, Ref s b)
splitP γ = do
  α <- newRef
  β <- newRef
  pTable %= (Split γ α β :)
  return (α, β)

type Forward = Map Int Dynamic

fValue :: Ref s a -> Lens' Forward (Maybe a)
fValue (Ref i) = lens
                 (\m -> unsafe <$> Map.lookup i m)
                 (\m x -> case x of
                           Just z -> Map.insert i (Dynamic z) m
                           Nothing -> Map.delete i m)

type Backward = Map Int Dynamic

bValue :: Ref s a -> Lens' Backward (Maybe (D a))
bValue (Ref i) = lens
                 (\m -> unsafe <$> Map.lookup i m)
                 (\m x -> case x of
                           Just z -> Map.insert i (Dynamic z) m
                           Nothing -> Map.delete i m)

type K m a b = D b -> m (D a)

data Back s m where
  SingleB :: (Additive m (D a)) => Ref s a -> Ref s b -> K m a b -> Back s m
  JoinB :: (Additive m (D a), Additive m (D b)) => Ref s a -> Ref s b -> Ref s (a,b) -> Back s m
  SplitB :: (Additive m (D a), Additive m (D b)) => Ref s (a,b) -> Ref s a -> Ref s b -> Back s m

unsafe :: Dynamic -> b
unsafe (Dynamic a) = unsafeCoerce a

traverseBack :: (Traversable t, Applicative f) => (a -> f b) -> t a -> f (t b)
traverseBack f = forwards . traverse (Backwards . f)

plusMaybe :: (Applicative m, Additive m a) => Maybe a -> Maybe a -> m (Maybe a)
plusMaybe (Just x) (Just y) = Just <$> plus x y
plusMaybe (Just x) Nothing = pure (Just x)
plusMaybe Nothing (Just y) = pure (Just y)
plusMaybe Nothing Nothing = pure Nothing

runPointed :: forall m a b. (Additive m (D a), Monad m) => (forall s. Ref s a -> Pointed s m (Ref s b)) -> Diff m a b
runPointed k = Diff run
  where
    ref0 = Ref 0 :: Ref s a
    (finalref, PData _ sources') = runState (k (Ref 0)) (PData 1 [])
    sources = reverse sources'
    run a = do
      let
        go :: Source s m -> StateT Forward m (Back s m)
        go (Single i j network) = do
          Just a <- use (fValue i)
          (b, k) <- lift $ runDiff network a
          fValue j .= Just b
          return (SingleB i j k)
        go (Join i j k) = do
          Just a <- use (fValue i)
          Just b <- use (fValue j)
          fValue k .= Just (a,b)
          return (JoinB i j k)
        go (Split γ α β) = do
          Just (a,b) <- use (fValue γ)
          fValue α .= Just a
          fValue β .= Just b
          return (SplitB γ α β)

      (back, values) <- runStateT (traverse go sources) (Map.singleton 0 (Dynamic a))
      let
        backward db = do
          let goback :: Back s m -> StateT Backward m ()
              goback (SingleB i j k) = do
                may_db <- use (bValue j)
                da <- case may_db of
                      Just db -> Just <$> lift (k db)
                      Nothing -> pure Nothing
                da_old <- use (bValue i)
                da_new <- lift (plusMaybe da da_old)
                bValue i .= da_new
                return ()

              goback (JoinB i j k) = do
                may_dc <- use (bValue k)
                (da,db) <- case may_dc of
                      Just (a,b) -> pure (Just a, Just b)
                      Nothing -> pure (Nothing, Nothing)

                da_old <- use (bValue i)
                da_new <- lift (plusMaybe da_old da)
                bValue i .= da_new

                db_old <- use (bValue j)
                db_new <- lift (plusMaybe db_old db)
                bValue j .= db_new

                return ()

              goback (SplitB γ α β) = do
                da <- maybe (lift zero) pure =<< use (bValue α)
                db <- maybe (lift zero) pure =<< use (bValue β)
                dc_old <- use (bValue γ)
                dc_new <- lift (plusMaybe dc_old (Just (da,db)))
                bValue γ .= dc_new
                return ()

          bmap <- execStateT (traverseBack goback back) (Map.singleton (getRef finalref) (Dynamic db))
          let may_da = view (bValue ref0) bmap
          case may_da of
           Just da -> return da
           Nothing -> zero

      let Just b = view (fValue finalref) values
      return (b, backward)
