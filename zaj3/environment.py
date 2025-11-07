import math
import time
from direct.showbase.ShowBase import ShowBase
import random
import sys
from direct.gui.OnscreenText import OnscreenText

from panda3d.core import (
    AmbientLight, DirectionalLight, Vec3, Vec4, Point3,
    CollisionTraverser, CollisionHandlerPusher,
    CollisionNode, CollisionSphere, CollisionBox,
    BitMask32, TransparencyAttrib, ClockObject
    , TextNode
)
import copy


class BallEnvironment(ShowBase):
    def __init__(self, n_balls=10, box_size=6.0, ball_radius=0.3):
        super().__init__()

        self.ball_radius = ball_radius

        # Basic scene: camera & lighting
        self.disableMouse()
        self.camera.setPos(-box_size * 1.0, -box_size * 2.0, box_size * 1.2)
        self.camera.lookAt(0, 0, 0)

        al = AmbientLight("ambient")
        al.setColor(Vec4(0.4, 0.4, 0.4, 1))
        al_np = self.render.attachNewNode(al)
        self.render.setLight(al_np)

        dl = DirectionalLight("dir")
        dl.setColor(Vec4(0.9, 0.9, 0.8, 1))
        dl_np = self.render.attachNewNode(dl)
        dl_np.setHpr(-30, -30, 0)
        self.render.setLight(dl_np)

        self.box_half = box_size / 2.0
        
        # create box visualization
        box = self.loader.loadModel("models/box")
        box.reparentTo(self.render)
        box.setScale(self.box_half*2)
        box.setPos(-self.box_half, -self.box_half, -self.box_half)
        box.setColor(1, 1, 1, 0.1)
        box.setTransparency(TransparencyAttrib.MAlpha)

        # Create balls
        self.balls = []
        for i in range(n_balls):
            # visual sphere
            sphere = self.loader.loadModel("models/misc/sphere")
            sphere.reparentTo(self.render)
            sphere.setScale(ball_radius)
            # random position inside the box (keep inside by margin)
            margin = ball_radius + 0.01
            x = random.uniform(-self.box_half + margin, self.box_half - margin)
            y = random.uniform(-self.box_half + margin, self.box_half - margin)
            z = random.uniform(-self.box_half + margin, self.box_half - margin)
            sphere.setPos(x, y, z)
            sphere.setColor(1, 1, 1, 1.0)


            # give a random initial velocity
            vel = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-0.5, 0.5))
            # normalize a bit and scale
            if vel.length() == 0:
                vel = Vec3(0.1, 0.2, 0.05)
            vel.normalize()
            vel *= random.uniform(1.0, 2.5)

            self.balls.append({"node": sphere, "vel": vel})

        # Task to move balls each frame and run collision traversal
        self.taskMgr.add(self.update, "update")

        # on-screen text that can be changed at runtime
        self.display_text = ""
        self.screen_text = OnscreenText(
            text=self.display_text,
            pos=(-1.3, 0.9),
            scale=0.07,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=True,
        )

        # Close on escape
        self.accept("escape", sys.exit)

    def update(self, task):
        dt = ClockObject.getGlobalClock().getDt()
        for b in self.balls:
            np = b["node"]
            np.setPos(np.getPos() + b["vel"] * dt)
            np.setColor(1, 1, np.getY(), 1.0)

        for b in self.balls:
            p = b["node"].getPos()
            v = b["vel"]
            # reflect velocity component if beyond extents (very small corrective buffer)
            buf = 1e-3
            if p.x >= self.box_half - buf and v.x > 0:
                v.x *= -1
            if p.x <= -self.box_half + buf and v.x < 0:
                v.x *= -1
            if p.y >= self.box_half - buf and v.y > 0:
                v.y *= -1
            if p.y <= -self.box_half + buf and v.y < 0:
                v.y *= -1
            if p.z >= self.box_half - buf and v.z > 0:
                v.z *= -1
            if p.z <= -self.box_half + buf and v.z < 0:
                v.z *= -1
            b["vel"] = v

        self.clearDisplayText()
        #self.detectCollisionsBruteForce()
        self.detectCollisionsSweepAndPrune()
        self.detectCollisionsBoundingBoxHierarchy()

        # keep the on-screen text in sync with `self.display_text`
        if hasattr(self, "screen_text") and self.screen_text:
            self.screen_text.setText(self.display_text)

        return task.cont
    
    def detectCollisionsBruteForce(self):
        counter = 0
        checks = 0
        for i in range(len(self.balls)):
            b1 = self.balls[i]
            p1 = b1["node"].getPos()
            r1 = b1["node"].getScale().x
            v1 = b1["vel"]
            for j in range(i + 1, len(self.balls)):
                b2 = self.balls[j]
                p2 = b2["node"].getPos()
                r2 = b2["node"].getScale().x
                v2 = b2["vel"]

                # check distance
                dist = (p2 - p1).length()
                checks += 1
                if dist < (r1 + r2):
                    counter += 1
        self.add_to_display_text(f"Naive\n:Collisions: {counter}\nChecks: {checks}\n")

    def detectCollisionsSweepAndPrune(self):
        # Step 1: Sort the balls along the x-axis
        self.balls.sort(key=lambda b: b["node"].getX())
        counter = 0
        checks = 0
        n = len(self.balls)

        for i in range(n):
            b1 = self.balls[i]
            p1 = b1["node"].getPos()
            r1 = b1["node"].getScale().x
            v1 = b1["vel"]
            for j in range(i + 1, n):
                b2 = self.balls[j]
                p2 = b2["node"].getPos()
                r2 = b2["node"].getScale().x

                # Since the list is sorted by x, we can break early
                if p2.x - r2 > p1.x + r1:
                    break

                # check distance
                dist = (p2 - p1).length()
                checks += 1
                if dist < (r1 + r2):
                    counter += 1

        self.add_to_display_text(f"S&P:\nCollisions: {counter}\nChecks: {checks}\n")

    def detectCollisionsBoundingBoxHierarchy(self):
        checks = 0
        collisions = 0
        def expandBits(v):
            v = (v * 0x00010001) & 0xFF0000FF
            v = (v * 0x00000101) & 0x0F00F00F
            v = (v * 0x00000011) & 0xC30C30C3
            v = (v * 0x00000005) & 0x49249249
            return v

        def calculateMortonCode(x, y, z):
            def normalizeCoord(val):
                return (val + self.box_half) / (self.box_half * 2)

            x = normalizeCoord(x)
            y = normalizeCoord(y)
            z = normalizeCoord(z)

            # Clamp to ensure they're in [0,1]
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            z = min(max(z, 0.0), 1.0)

            # Scale to range [0, 1023] for 10-bit encoding
            x = min(math.floor(x * 1023), 1023)
            y = min(math.floor(y * 1023), 1023)
            z = min(math.floor(z * 1023), 1023)

            # Insert zeros between bits (3D Morton code)
            xx = expandBits(x)
            yy = expandBits(y)
            zz = expandBits(z)

            # Interleave the bits
            return xx | (yy << 1) | (zz << 2)

        def getAABB(b):
            p = b["node"].getPos()
            r = b["node"].getScale().x
            return {
                "min": Vec3(p.x - r, p.y - r, p.z - r),
                "max": Vec3(p.x + r, p.y + r, p.z + r)
            }

        def checkAABBCollision(aabb1, aabb2):
            return (aabb1["min"].x <= aabb2["max"].x and aabb1["max"].x >= aabb2["min"].x) and \
                   (aabb1["min"].y <= aabb2["max"].y and aabb1["max"].y >= aabb2["min"].y) and \
                   (aabb1["min"].z <= aabb2["max"].z and aabb1["max"].z >= aabb2["min"].z)

        def checkCollision(b1, b2):
            b1Pos = b1["node"].getPos()
            b2Pos = b2["node"].getPos()
            return (b1Pos - b2Pos).length() < (b1["node"].getScale().x + b2["node"].getScale().x)
        
        class BVHNode:
            def __init__(self):
                self.left = None
                self.right = None
                self.ballRef = None  # Only leaf nodes have valid ballRefs
                self.aabb = {
                    "min": Vec3(float("inf"), float("inf"), float("inf")),
                    "max": Vec3(float("-inf"), float("-inf"), float("-inf"))
                }

            def isLeaf(self):
                return self.ballRef is not None

        def getSplitPos(lst, begin, end):
            # Simple middle split strategy
            return (begin + end) // 2

        def createLeaf(ball):
            node = BVHNode()
            node.ballRef = ball

            # Set AABB from ball
            aabb = getAABB(ball)
            node.aabb = copy.deepcopy(aabb)

            return node

        def createSubTree(lst, begin, end, balls):
            if begin == end:
                return createLeaf(balls[lst[begin]["id"]])
            else:
                m = getSplitPos(lst, begin, end)
                node = BVHNode()

                node.left = createSubTree(lst, begin, m, balls)
                node.right = createSubTree(lst, m + 1, end, balls)

                # Update node's AABB to encompass children's AABBs
                node.aabb['min'].x = min(node.left.aabb['min'].x, node.right.aabb['min'].x)
                node.aabb['min'].y = min(node.left.aabb['min'].y, node.right.aabb['min'].y)
                node.aabb['min'].z = min(node.left.aabb['min'].z, node.right.aabb['min'].z)

                node.aabb['max'].x = max(node.left.aabb['max'].x, node.right.aabb['max'].x)
                node.aabb['max'].y = max(node.left.aabb['max'].y, node.right.aabb['max'].y)
                node.aabb['max'].z = max(node.left.aabb['max'].z, node.right.aabb['max'].z)

                return node
            
        

        def createTree(balls):
            # Create list of ball IDs with their Morton codes
            lst = []
            for i in range(len(balls)):
                ball = balls[i]["node"]
                center = ball.getPos()
                mortonCode = calculateMortonCode(center.x, center.y, center.z)
                lst.append({"id": i, "mortonCode": mortonCode})
            

            # Sort by Morton code for spatial locality
            lst.sort(key=lambda x: x["mortonCode"])

            # Create the BVH tree recursively
            return createSubTree(lst, 0, len(lst) - 1, balls)

        def findCollisions(balls, root):
            for ball in balls:
                findCollisionsRec(ball, root)

        def findCollisionsRec(ball, node):
            nonlocal collisions, checks
            checks += 1
            # If this box's AABB doesn't intersect with the node's AABB, return
            if not checkAABBCollision(getAABB(ball), node.aabb):
                return

            # If this is a leaf node
            if node.isLeaf():
                # Don't check collisions with self
                if node.ballRef != ball:
                    # Check for actual collision between boxes
                    if checkCollision(node.ballRef, ball):
                        collisions += 1
                return

            # If this is not a leaf node, recurse through children
            findCollisionsRec(ball, node.left)
            findCollisionsRec(ball, node.right)

        t = time.time()
        root = createTree(self.balls)
        time_build_tree = time.time() - t
        t = time.time()
        findCollisions(self.balls, root)
        time_collision_detection = time.time() - t

        self.add_to_display_text(f"BBVH:\nCollisions: {collisions//2}\nChecks: {checks}\nTime Build: {time_build_tree:.4f}s\nTime Detect: {time_collision_detection:.4f}s\n")


    def add_to_display_text(self, text: str):
        """Set the on-screen display text. Safe to call from anywhere (including update).

        Example: self.add_to_display_text(f"Balls: {len(self.balls)}")
        """
        self.display_text += str(text)
        if hasattr(self, "screen_text") and self.screen_text:
            self.screen_text.setText(self.display_text)
    
    def clearDisplayText(self):
        """Clear the on-screen display text."""
        self.display_text = ""
        if hasattr(self, "screen_text") and self.screen_text:
            self.screen_text.setText(self.display_text)


if __name__ == "__main__":
    app = BallEnvironment(n_balls=1000, box_size=200.0, ball_radius=1)
    app.run()