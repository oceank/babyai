import gym
from gym_minigrid.envs import Key, Ball, Box
from .verifier import *
from .levelgen import *
#from gym_minigrid.minigrid import WorldObj, COLOR_TO_IDX, OBJECT_TO_IDX

class Level_GoToRedBlueBall(RoomGridLevel):
    """
    Go to the red ball or to the blue ball.
    There is exactly one red or blue ball, and some distractors.
    The distractors are guaranteed not to be red or blue balls.
    Language is not required to solve this level.
    """

    def __init__(self, room_size=8, num_dists=7, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()

        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # Ensure there is only one red or blue ball
        for dist in dists:
            if dist.type == 'ball' and (dist.color == 'blue' or dist.color == 'red'):
                raise RejectSampling('can only have one blue or red ball')

        color = self._rand_elem(['red', 'blue'])
        obj, _ = self.add_object(0, 0, 'ball', color)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_OpenRedDoor(RoomGridLevel):
    """
    Go to the red door
    (always unlocked, in the current room)
    Note: this level is intentionally meant for debugging and is
    intentionally kept very simple.
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=5,
            seed=seed
        )

    def gen_mission(self):
        obj, _ = self.add_door(0, 0, 0, 'red', locked=False)
        self.place_agent(0, 0)
        self.instrs = OpenInstr(ObjDesc('door', 'red'))


class Level_OpenDoor(RoomGridLevel):
    """
    Go to the door
    The door to open is given by its color or by its location.
    (always unlocked, in the current room)
    """

    def __init__(
        self,
        debug=False,
        select_by=None,
        seed=None
    ):
        self.select_by = select_by
        self.debug = debug
        super().__init__(seed=seed)

    def gen_mission(self):
        door_colors = self._rand_subset(COLOR_NAMES, 4)
        objs = []

        for i, color in enumerate(door_colors):
            obj, _ = self.add_door(1, 1, door_idx=i, color=color, locked=False)
            objs.append(obj)

        select_by = self.select_by
        if select_by is None:
            select_by = self._rand_elem(["color", "loc"])
        if select_by == "color":
            object = ObjDesc(objs[0].type, color=objs[0].color)
        elif select_by == "loc":
            object = ObjDesc(objs[0].type, loc=self._rand_elem(LOC_NAMES))

        self.place_agent(1, 1)
        self.instrs = OpenInstr(object, strict=self.debug)


class Level_OpenDoorDebug(Level_OpenDoor):
    """
    Same as OpenDoor but the level stops when any door is opened
    """

    def __init__(
        self,
        select_by=None,
        seed=None
    ):
        super().__init__(select_by=select_by, debug=True, seed=seed)


class Level_OpenDoorColor(Level_OpenDoor):
    """
    Go to the door
    The door is selected by color.
    (always unlocked, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            select_by="color",
            seed=seed
        )


#class Level_OpenDoorColorDebug(Level_OpenDoorColor, Level_OpenDoorDebug):
    """
    Same as OpenDoorColor but the level stops when any door is opened
    """
#    pass


class Level_OpenDoorLoc(Level_OpenDoor):
    """
    Go to the door
    The door is selected by location.
    (always unlocked, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            select_by="loc",
            seed=seed
        )


class Level_GoToDoor(RoomGridLevel):
    """
    Go to a door
    (of a given color, in the current room)
    No distractors, no language variation
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            seed=seed
        )

    def gen_mission(self):
        objs = []
        for _ in range(4):
            door, _ = self.add_door(1, 1)
            objs.append(door)
        self.place_agent(1, 1)

        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc('door', obj.color))


class Level_GoToObjDoor(RoomGridLevel):
    """
    Go to an object or door
    (of a given type and color, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=8,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent(1, 1)
        objs = self.add_distractors(1, 1, num_distractors=8, all_unique=False)

        for _ in range(4):
            door, _ = self.add_door(1, 1)
            objs.append(door)

        self.check_objs_reachable()

        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_ActionObjDoor(RoomGridLevel):
    """
    [pick up an object] or
    [go to an object or door] or
    [open a door]
    (in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            seed=seed
        )

    def gen_mission(self):
        objs = self.add_distractors(1, 1, num_distractors=5)
        for _ in range(4):
            door, _ = self.add_door(1, 1, locked=False)
            objs.append(door)

        self.place_agent(1, 1)

        obj = self._rand_elem(objs)
        desc = ObjDesc(obj.type, obj.color)

        if obj.type == 'door':
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = OpenInstr(desc)
        else:
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = PickupInstr(desc)


class Level_UnlockLocal(RoomGridLevel):
    """
    Fetch a key and unlock a door
    (in the current room)
    """

    def __init__(self, distractors=False, seed=None):
        self.distractors = distractors
        super().__init__(seed=seed)

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)
        self.add_object(1, 1, 'key', door.color)
        if self.distractors:
            self.add_distractors(1, 1, num_distractors=3)
        self.place_agent(1, 1)

        self.instrs = OpenInstr(ObjDesc(door.type))


class Level_UnlockLocalDist(Level_UnlockLocal):
    """
    Fetch a key and unlock a door
    (in the current room, with distractors)
    """

    def __init__(self, seed=None):
        super().__init__(distractors=True, seed=seed)


class Level_KeyInBox(RoomGridLevel):
    """
    Unlock a door. Key is in a box (in the current room).
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)

        # Put the key in the box, then place the box in the room
        key = Key(door.color)
        box = Box(self._rand_color(), key)
        self.place_in_room(1, 1, box)

        self.place_agent(1, 1)

        self.instrs = OpenInstr(ObjDesc(door.type))

        self.sub_goals = [
            {"instr": GoToInstr(ObjDesc(box.type, box.color))},
            #{"instr": PickupInstr(ObjDesc(key.type, key.color))},
            {"instr": GoToInstr(ObjDesc(door.type))},
            {"instr": OpenInstr(ObjDesc(door.type))}
            ]

class Level_UnlockPickup(RoomGridLevel):
    """
    Unlock a door, then pick up a box in another room
    """

    def __init__(self, distractors=False, seed=None):
        self.distractors = distractors

        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', door.color)
        if self.distractors:
            self.add_distractors(num_distractors=4)

        self.place_agent(0, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_UnlockPickupDist(Level_UnlockPickup):
    """
    Unlock a door, then pick up an object in another room
    (with distractors)
    """

    def __init__(self, seed=None):
        super().__init__(distractors=True, seed=seed)


class Level_BlockedUnlockPickup(RoomGridLevel):
    """
    Unlock a door blocked by a ball, then pick up a box
    in another room
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=16*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, pos = self.add_door(0, 0, 0, locked=True)
        # Block the door with a ball
        color = self._rand_color()
        self.grid.set(pos[0]-1, pos[1], Ball(color))
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', door.color)

        self.place_agent(0, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_UnlockToUnlock(RoomGridLevel):
    """
    Unlock a door A that requires to unlock a door B before
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            max_steps=30*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, 2)

        # Add a door of color A connecting left and middle room
        self.add_door(0, 0, door_idx=0, color=colors[0], locked=True)

        # Add a key of color A in the room on the right
        self.add_object(2, 0, kind="key", color=colors[0])

        # Add a door of color B connecting middle and right room
        self.add_door(1, 0, door_idx=0, color=colors[1], locked=True)

        # Add a key of color B in the middle room
        self.add_object(1, 0, kind="key", color=colors[1])

        obj, _ = self.add_object(0, 0, kind="ball")

        self.place_agent(1, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_PickupDist(RoomGridLevel):
    """
    Pick up an object
    The object to pick up is given by its type only, or
    by its color, or by its type and color.
    (in the current room, with distractors)
    """

    def __init__(self, debug=False, seed=None):
        self.debug = debug
        super().__init__(
            num_rows = 1,
            num_cols = 1,
            room_size=7,
            seed=seed
        )

    def gen_mission(self):
        # Add 5 random objects in the room
        objs = self.add_distractors(num_distractors=5)
        self.place_agent(0, 0)
        obj = self._rand_elem(objs)
        type = obj.type
        color = obj.color

        select_by = self._rand_elem(["type", "color", "both"])
        if select_by == "color":
            type = None
        elif select_by == "type":
            color = None

        self.instrs = PickupInstr(ObjDesc(type, color), strict=self.debug)


class Level_PickupDistDebug(Level_PickupDist):
    """
    Same as PickupDist but the level stops when any object is picked
    """

    def __init__(self, seed=None):
        super().__init__(
            debug=True,
            seed=seed
        )


class Level_PickupAbove(RoomGridLevel):
    """
    Pick up an object (in the room above)
    This task requires to use the compass to be solved effectively.
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to the top-middle room
        obj, pos = self.add_object(1, 0)
        # Make sure the two rooms are directly connected
        self.add_door(1, 1, 3, locked=False)
        self.place_agent(1, 1)
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_OpenTwoDoors(RoomGridLevel):
    """
    Open door X, then open door Y
    The two doors are facing opposite directions, so that the agent
    Can't see whether the door behind him is open.
    This task requires memory (recurrent policy) to be solved effectively.
    """

    def __init__(self,
        first_color=None,
        second_color=None,
        strict=False,
        seed=None
    ):
        self.first_color = first_color
        self.second_color = second_color
        self.strict = strict

        room_size = 6
        super().__init__(
            room_size=room_size,
            max_steps=20*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, 2)

        first_color = self.first_color
        if first_color is None:
            first_color = colors[0]
        second_color = self.second_color
        if second_color is None:
            second_color = colors[1]

        door1, _ = self.add_door(1, 1, 2, color=first_color, locked=False)
        door2, _ = self.add_door(1, 1, 0, color=second_color, locked=False)

        self.place_agent(1, 1)

        self.instrs = BeforeInstr(
            OpenInstr(ObjDesc(door1.type, door1.color), strict=self.strict),
            OpenInstr(ObjDesc(door2.type, door2.color))
        )


class Level_OpenTwoDoorsDebug(Level_OpenTwoDoors):
    """
    Same as OpenTwoDoors but the level stops when the second door is opened
    """

    def __init__(self,
        first_color=None,
        second_color=None,
        seed=None
    ):
        super().__init__(
            first_color,
            second_color,
            strict=True,
            seed=seed
        )


class Level_OpenRedBlueDoors(Level_OpenTwoDoors):
    """
    Open red door, then open blue door
    The two doors are facing opposite directions, so that the agent
    Can't see whether the door behind him is open.
    This task requires memory (recurrent policy) to be solved effectively.
    """

    def __init__(self, seed=None):
        super().__init__(
            first_color="red",
            second_color="blue",
            seed=seed
        )


class Level_OpenRedBlueDoorsDebug(Level_OpenTwoDoorsDebug):
    """
    Same as OpenRedBlueDoors but the level stops when the blue door is opened
    """

    def __init__(self, seed=None):
        super().__init__(
            first_color="red",
            second_color="blue",
            seed=seed
        )


class Level_FindObjS5(RoomGridLevel):
    """
    Pick up an object (in a random room)
    Rooms have a size of 5
    This level requires potentially exhaustive exploration
    """

    def __init__(self, room_size=5, seed=None):
        super().__init__(
            room_size=room_size,
            max_steps=20*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to a random room
        i = self._rand_int(0, self.num_rows)
        j = self._rand_int(0, self.num_cols)
        obj, _ = self.add_object(i, j)
        self.place_agent(1, 1)
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_FindObjS6(Level_FindObjS5):
    """
    Same as the FindObjS5 level, but rooms have a size of 6
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            seed=seed
        )


class Level_FindObjS7(Level_FindObjS5):
    """
    Same as the FindObjS5 level, but rooms have a size of 7
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            seed=seed
        )


class KeyCorridor(RoomGridLevel):
    """
    A ball is behind a locked door, the key is placed in a
    random room.
    """

    def __init__(
        self,
        num_rows=3,
        obj_type="ball",
        room_size=6,
        seed=None
    ):
        self.obj_type = obj_type

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            max_steps=30*room_size**2,
            seed=seed,
        )

    def gen_mission(self):
        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        # Add a locked door on the bottom right
        # Add an object behind the locked door
        room_idx = self._rand_int(0, self.num_rows)
        door, _ = self.add_door(2, room_idx, 2, locked=True)
        obj, _ = self.add_object(2, room_idx, kind=self.obj_type)

        # Add a key in a random room on the left side
        self.add_object(0, self._rand_int(0, self.num_rows), 'key', door.color)

        # Place the agent in the middle
        self.place_agent(1, self.num_rows // 2)

        # Make sure all rooms are accessible
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_KeyCorridorS3R1(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=3,
            num_rows=1,
            seed=seed
        )

class Level_KeyCorridorS3R2(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=3,
            num_rows=2,
            seed=seed
        )

class Level_KeyCorridorS3R3(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=3,
            num_rows=3,
            seed=seed
        )

class Level_KeyCorridorS4R3(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=4,
            num_rows=3,
            seed=seed
        )

class Level_KeyCorridorS5R3(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            num_rows=3,
            seed=seed
        )

class Level_KeyCorridorS6R3(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            num_rows=3,
            seed=seed
        )

class Level_1RoomS8(RoomGridLevel):
    """
    Pick up the ball
    Rooms have a size of 8
    """

    def __init__(self, room_size=8, seed=None):
        super().__init__(
            room_size=room_size,
            num_rows=1,
            num_cols=1,
            seed=seed
        )

    def gen_mission(self):
        obj, _ = self.add_object(0, 0, kind="ball")
        self.place_agent()
        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_1RoomS12(Level_1RoomS8):
    """
    Pick up the ball
    Rooms have a size of 12
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=12,
            seed=seed
        )


class Level_1RoomS16(Level_1RoomS8):
    """
    Pick up the ball
    Rooms have a size of 16
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=16,
            seed=seed
        )


class Level_1RoomS20(Level_1RoomS8):
    """
    Pick up the ball
    Rooms have a size of 20
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=20,
            seed=seed
        )


class PutNext(RoomGridLevel):
    """
    Task of the form: move the A next to the B and the C next to the D.
    This task is structured to have a very large number of possible
    instructions.
    """

    def __init__(
        self,
        room_size,
        objs_per_room,
        start_carrying=False,
        seed=None
    ):
        assert room_size >= 4
        assert objs_per_room <= 9
        self.objs_per_room = objs_per_room
        self.start_carrying = start_carrying

        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent(0, 0)

        # Add objects to both the left and right rooms
        # so that we know that we have two non-adjacent set of objects
        objs_l = self.add_distractors(0, 0, self.objs_per_room)
        objs_r = self.add_distractors(1, 0, self.objs_per_room)

        # Remove the wall between the two rooms
        self.remove_wall(0, 0, 0)

        # Select objects from both subsets
        a = self._rand_elem(objs_l)
        b = self._rand_elem(objs_r)

        # Randomly flip the object to be moved
        if self._rand_bool():
            t = a
            a = b
            b = t

        self.obj_a = a

        self.instrs = PutNextInstr(
            ObjDesc(a.type, a.color),
            ObjDesc(b.type, b.color)
        )

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # If the agent starts off carrying the object
        if self.start_carrying:
            self.grid.set(*self.obj_a.init_pos, None)
            self.carrying = self.obj_a

        return obs


class Level_PutNextS4N1(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=4,
            objs_per_room=1,
            seed=seed
        )


class Level_PutNextS5N1(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            objs_per_room=1,
            seed=seed
        )


class Level_PutNextS5N2(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            objs_per_room=2,
            seed=seed
        )


class Level_PutNextS6N3(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            objs_per_room=3,
            seed=seed
        )


class Level_PutNextS7N4(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            objs_per_room=4,
            seed=seed
        )


class Level_PutNextS5N2Carrying(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            objs_per_room=2,
            start_carrying=True,
            seed=seed
        )


class Level_PutNextS6N3Carrying(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            objs_per_room=3,
            start_carrying=True,
            seed=seed
        )


class Level_PutNextS7N4Carrying(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            objs_per_room=4,
            start_carrying=True,
            seed=seed
        )


class MoveTwoAcross(RoomGridLevel):
    """
    Task of the form: move the A next to the B and the C next to the D.
    This task is structured to have a very large number of possible
    instructions.
    """

    def __init__(
        self,
        room_size,
        objs_per_room,
        seed=None
    ):
        assert objs_per_room <= 9
        self.objs_per_room = objs_per_room

        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=16*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent(0, 0)

        # Add objects to both the left and right rooms
        # so that we know that we have two non-adjacent set of objects
        objs_l = self.add_distractors(0, 0, self.objs_per_room)
        objs_r = self.add_distractors(1, 0, self.objs_per_room)

        # Remove the wall between the two rooms
        self.remove_wall(0, 0, 0)

        # Select objects from both subsets
        objs_l = self._rand_subset(objs_l, 2)
        objs_r = self._rand_subset(objs_r, 2)
        a = objs_l[0]
        b = objs_r[0]
        c = objs_r[1]
        d = objs_l[1]

        self.instrs = BeforeInstr(
            PutNextInstr(ObjDesc(a.type, a.color), ObjDesc(b.type, b.color)),
            PutNextInstr(ObjDesc(c.type, c.color), ObjDesc(d.type, d.color))
        )


class Level_MoveTwoAcrossS5N2(MoveTwoAcross):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            objs_per_room=2,
            seed=seed
        )


class Level_MoveTwoAcrossS8N9(MoveTwoAcross):
    def __init__(self, seed=None):
        super().__init__(
            room_size=8,
            objs_per_room=9,
            seed=seed
        )


class OpenDoorsOrder(RoomGridLevel):
    """
    Open one or two doors in the order specified.
    """

    def __init__(
        self,
        num_doors,
        debug=False,
        seed=None
    ):
        assert num_doors >= 2
        self.num_doors = num_doors
        self.debug = debug

        room_size = 6
        super().__init__(
            room_size=room_size,
            max_steps=20*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, self.num_doors)
        doors = []
        for i in range(self.num_doors):
            door, _ = self.add_door(1, 1, color=colors[i], locked=False)
            doors.append(door)
        self.place_agent(1, 1)

        door1, door2 = self._rand_subset(doors, 2)
        desc1 = ObjDesc(door1.type, door1.color)
        desc2 = ObjDesc(door2.type, door2.color)

        mode = self._rand_int(0, 3)
        if mode == 0:
            self.instrs = OpenInstr(desc1, strict=self.debug)
        elif mode == 1:
            self.instrs = BeforeInstr(OpenInstr(desc1, strict=self.debug), OpenInstr(desc2, strict=self.debug))
        elif mode == 2:
            self.instrs = AfterInstr(OpenInstr(desc1, strict=self.debug), OpenInstr(desc2, strict=self.debug))
        else:
            assert False

class Level_OpenDoorsOrderN2(OpenDoorsOrder):
    def __init__(self, seed=None):
        super().__init__(
            num_doors=2,
            seed=seed
        )


class Level_OpenDoorsOrderN4(OpenDoorsOrder):
    def __init__(self, seed=None):
        super().__init__(
            num_doors=4,
            seed=seed
        )


class Level_OpenDoorsOrderN2Debug(OpenDoorsOrder):
    def __init__(self, seed=None):
        super().__init__(
            num_doors=2,
            debug=True,
            seed=seed
        )


class Level_OpenDoorsOrderN4Debug(OpenDoorsOrder):
    def __init__(self, seed=None):
        super().__init__(
            num_doors=4,
            debug=True,
            seed=seed
        )

# Environments for high-level and low-level goals
## Low-level tasks that are associated with two rooms
class Level_PickupKeyLocalR2(RoomGridLevel):
    """
    Fetch a key in the current room
    """

    def __init__(self, room_size=8, num_distractors=0, seed=None):
        self.num_distractors = num_distractors

        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        door, _ = self.add_door(0, 0, 0, locked=True)

        key = Key(door.color)
        self.place_in_room(0, 0, key)

        if self.num_distractors > 0:
            self.add_distractors(0, 0, num_distractors=self.num_distractors)

        self.place_agent(0, 0)

        self.instrs = PickupInstr(ObjDesc(key.type, key.color))

class Level_PickupKeyLocalR2Dist(Level_PickupKeyLocalR2):
    """
    Fetch a key in the current room
    """

    def __init__(self, room_size=8, seed=None):

        super().__init__(
            room_size=room_size,
            num_distractors=4,
            seed=seed
        )

class Level_UnlockLocalR2(RoomGridLevel):
    """
    Fetch a key and unlock a door
    (in the current room)
    """

    def __init__(self, room_size=8, num_distractors=0, seed=None, use_subgoals=False):
        self.num_distractors = num_distractors
        self.use_subgoals = use_subgoals

        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        door, _ = self.add_door(0, 0, 0, locked=True)

        key = Key(door.color)
        self.place_in_room(0, 0, key)

        if self.num_distractors > 0:
            self.add_distractors(0, 0, num_distractors=self.num_distractors)

        self.place_agent(0, 0)

        self.instrs = OpenInstr(ObjDesc(door.type, door.color))

        if self.use_subgoals:
            self.sub_goals = [
            {"instr": PickupInstr(ObjDesc(key.type, key.color))},
            {"instr": OpenInstr(ObjDesc(door.type, door.color))}
            ]

class Level_UnlockLocalR2Dist(Level_UnlockLocalR2):
    """
    Fetch a key and unlock a door
    (in the current room)
    """

    def __init__(self, room_size=8, num_distractors=4, seed=None, use_subgoals=False):

        super().__init__(
            room_size=room_size,
            num_distractors=num_distractors,
            seed=seed,
            use_subgoals=use_subgoals
        )

class Level_UnlockLocalR2SubGoal(Level_UnlockLocalR2):
    """
    Fetch a key and unlock a door
    (in the current room)
    """

    def __init__(self, room_size=8, num_distractors=0, seed=None, use_subgoals=True):
        super().__init__(
            room_size=room_size,
            num_distractors=num_distractors,
            seed=seed,
            use_subgoals=use_subgoals
        )

class Level_UnlockLocalR2DistSubGoal(Level_UnlockLocalR2):
    """
    Fetch a key and unlock a door
    (in the current room)
    """

    def __init__(self, room_size=8, num_distractors=4, seed=None, use_subgoals=True):

        super().__init__(
            room_size=room_size,
            num_distractors=num_distractors,
            seed=seed,
            use_subgoals=use_subgoals
        )

class Level_OpenDoorLocalR2(RoomGridLevel):
    """
    Open a closed door in the current room
    """

    def __init__(self, room_size=8, num_distractors=0, seed=None):
        self.num_distractors = num_distractors

        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        door, _ = self.add_door(0, 0, locked=False)

        if self.num_distractors > 0:
            self.add_distractors(0, 0, num_distractors=self.num_distractors)

        self.place_agent(0, 0)

        self.instrs = OpenInstr(ObjDesc(door.type, door.color))

class Level_OpenDoorLocalR2Dist(Level_OpenDoorLocalR2):
    """
    Open a closed door that is in the current room.
    And there are some distractors in the current room.
    """

    def __init__(self, room_size=8, num_distractors=4, seed=None):

        super().__init__(
            room_size=room_size,
            num_distractors=num_distractors,
            seed=seed
        )

class Level_PassDoorLocalR2(RoomGridLevel):
    """
    Pass an opened door that is in the current room
    """

    def __init__(self, room_size=8, num_distractors=0, seed=None):
        self.num_distractors = num_distractors

        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        door, _ = self.add_door(0, 0, locked=False)
        door.is_open = True

        if self.num_distractors > 0:
            self.add_distractors(0, 0, num_distractors=self.num_distractors)

        self.place_agent(0, 0)

        self.instrs = PassInstr(ObjDesc(door.type, door.color))

class Level_PassDoorLocalR2Dist(Level_PassDoorLocalR2):
    """
    Pass an opened door that is in the current room.
    And there are some distractors in the current room.
    """

    def __init__(self,room_size=8, num_distractors=4, seed=None):
        super().__init__(
            room_size=room_size,
            num_distractors=num_distractors,
            seed=seed
        )

##  Low-level tasks in one room
class Level_OpenBoxLocalR1Dist(RoomGridLevel):
    """
    Open a box in the current room.
    """

    def __init__(self, room_size=8, num_distractors=4, seed=None):
        self.num_distractors = num_distractors
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors-1, all_unique=True)
        box_color = self._rand_elem(COLOR_NAMES)
        obj, _ = self.add_object(0, 0, 'box', box_color)

        self.place_agent(i=0, j=0)

        self.instrs = OpenBoxInstr(ObjDesc(obj.type, obj.color))

class Level_PickupLocalR1Dist(RoomGridLevel):
    """
    Fetch an object (key, ball or box) in the current room
    """

    def __init__(self, room_size=8, num_distractors=4, seed=None):
        self.num_distractors = num_distractors
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors, all_unique=True)
        obj = self._rand_elem(objs)

        self.place_agent(i=0, j=0)

        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))

class Level_GoToLocalR1Dist(RoomGridLevel):
    """
    Fetch an object (key, ball or box) in the current room
    """

    def __init__(self, room_size=8, num_distractors=4, seed=None):
        self.num_distractors = num_distractors
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors, all_unique=True)
        obj = self._rand_elem(objs)

        self.place_agent(i=0, j=0)

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

class Level_DropNextLocalR1Dist(RoomGridLevel):
    """
    Drop an object (key, ball or box) to an object
    """

    def __init__(self, room_size=8, num_distractors=4, seed=None):
        self.num_distractors = num_distractors
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors, all_unique=True)
        next_to_obj = self._rand_elem(objs)

        self.place_agent(i=0, j=0)

        carried_obj_color = self._rand_elem(COLOR_NAMES)
        carried_obj_type = self._rand_elem(['key', 'ball', 'box'])

        self.carrying = WorldObj.decode(OBJECT_TO_IDX[carried_obj_type], COLOR_TO_IDX[carried_obj_color], 0)
        self.carrying.cur_pos = np.array([-1, -1])

        self.instrs = DropNextInstr(
            obj_carried = ObjDesc(carried_obj_type, carried_obj_color),
            obj_fixed = ObjDesc(next_to_obj.type, next_to_obj.color),
            initial_carried_world_obj = self.carrying
        )

class Level_DropNextNothingLocalR1Dist(RoomGridLevel):
    """
    Drop a carried object and make sure it is not next to any other objects
    """

    def __init__(self, room_size=8, num_distractors=4, seed=None):
        self.num_distractors = num_distractors
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors, all_unique=True)

        self.place_agent(i=0, j=0)

        carried_obj_color = self._rand_elem(COLOR_NAMES)
        carried_obj_type = self._rand_elem(['key', 'ball', 'box'])

        self.carrying = WorldObj.decode(OBJECT_TO_IDX[carried_obj_type], COLOR_TO_IDX[carried_obj_color], 0)

        self.instrs = DropNextNothingInstr(
            initial_carried_world_obj = self.carrying,
            obj_to_drop = ObjDesc(carried_obj_type, carried_obj_color)
        )

# Environments For Low-level Tasks
class Level_ActionObjDoorR3(RoomGridLevel):
    """
    [pick up an object] or
    [go to an object or door] or
    [open a door]
    (in the current room)
    """

    def __init__(self, seed=None, door_locked=False, num_distractors=5):
        self.door_locked = door_locked
        self.num_distractors = num_distractors
        self.num_doors = 2
        super().__init__(
            room_size=8,
            num_rows=1,
            num_cols=3,
            agent_view_size=7,
            seed=seed
        )

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors)
        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)
            objs.append(door)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        obj = self._rand_elem(objs)
        desc = ObjDesc(obj.type, obj.color)

        if obj.type == 'door':
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = OpenInstr(desc)
        else:
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = PickupInstr(desc)


class Level_GoToLocalR3(Level_ActionObjDoorR3):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors)
        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)
            objs.append(door)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        obj = self._rand_elem(objs)
        desc = ObjDesc(obj.type, obj.color)
        self.instrs = GoToInstr(desc)

class Level_OpenBoxLocalR3(Level_ActionObjDoorR3):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        # add a box
        color = self._rand_elem(COLOR_NAMES)
        type = 'box'
        obj = (type, color)
        box, pos = self.add_object(1, 0, *obj)

        # add distractors
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors-1)
        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(box.type, box.color)
        self.instrs = OpenBoxInstr(desc)

class Level_OpenDoorLocalR3(Level_ActionObjDoorR3):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors)
        doors = []
        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)
            doors.append(door)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        door = self._rand_elem(doors)
        desc = ObjDesc(door.type, door.color)
        self.instrs = OpenInstr(desc)

class Level_PassDoorLocalR3(Level_ActionObjDoorR3):
    def __init__(self, seed=None):
        self.door_locked = False
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors)
        doors = []
        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, is_open=True, locked=self.door_locked)
            doors.append(door)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        door = self._rand_elem(doors)
        desc = ObjDesc(door.type, door.color)
        self.instrs = PassInstr(desc)

class Level_PickupLocalR3(Level_ActionObjDoorR3):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors)
        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        obj = self._rand_elem(objs)
        desc = ObjDesc(obj.type, obj.color)
        self.instrs = PickupInstr(desc)

class Level_DropNextToLocalR3(Level_ActionObjDoorR3):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors)
        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)
            objs.append(door)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        next_to_obj = self._rand_elem(objs)

        carried_obj_color = self._rand_elem(COLOR_NAMES)
        carried_obj_type = self._rand_elem(['key', 'ball', 'box'])

        self.carrying = WorldObj.decode(OBJECT_TO_IDX[carried_obj_type], COLOR_TO_IDX[carried_obj_color], 0)
        self.carrying.cur_pos = np.array([-1, -1])

        self.instrs = DropNextInstr(
            obj_carried = ObjDesc(carried_obj_type, carried_obj_color),
            obj_fixed = ObjDesc(next_to_obj.type, next_to_obj.color),
            initially_carried_world_obj = self.carrying
        )

class Level_DropNotNextToLocalR3(Level_ActionObjDoorR3):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors)
        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)
            objs.append(door)

        self.place_agent(i=1, j=0)
        
        # Make sure no unblocking is required
        self.check_objs_reachable()

        not_next_to_obj = self._rand_elem(objs)

        carried_obj_color = self._rand_elem(COLOR_NAMES)
        carried_obj_type = self._rand_elem(['key', 'ball', 'box'])

        self.carrying = WorldObj.decode(OBJECT_TO_IDX[carried_obj_type], COLOR_TO_IDX[carried_obj_color], 0)
        self.carrying.cur_pos = np.array([-1, -1])

        self.instrs = DropNotNextInstr(
            obj_carried = ObjDesc(carried_obj_type, carried_obj_color),
            obj_fixed = ObjDesc(not_next_to_obj.type, not_next_to_obj.color),
            initially_carried_world_obj = self.carrying
        )

class Level_DropNextNothingLocalR3(Level_ActionObjDoorR3):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors)
        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)
            objs.append(door)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        carried_obj_color = self._rand_elem(COLOR_NAMES)
        carried_obj_type = self._rand_elem(['key', 'ball', 'box'])

        self.carrying = WorldObj.decode(OBJECT_TO_IDX[carried_obj_type], COLOR_TO_IDX[carried_obj_color], 0)

        self.instrs = DropNextNothingInstr(
            initially_carried_world_obj = self.carrying,
            obj_to_drop = ObjDesc(carried_obj_type, carried_obj_color)
        )

## High-level tasks
### Two-Subgoal Task Group
class Level_UnlockLocalR3(Level_ActionObjDoorR3):
    def __init__(self, seed=None):
        super().__init__(seed=seed, door_locked=True)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        doors = []
        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)
            doors.append(door)

        target_door = self._rand_elem(doors)
        key, _ = self.add_object(1, 0, 'key', target_door.color)
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors-1)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(target_door.type, target_door.color)
        self.instrs = OpenInstr(desc)

class Level_PickupObjInBoxLocalR3(Level_ActionObjDoorR3):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        box = Box(self._rand_color())
        self.place_in_room(1, 0, box)

        # Add distractors and put one in the above box
        #   the box and the hidden object together are considered as one distractor
        #   since only one of them will be visible at a time
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors)
        hidden_obj = self._rand_elem(objs)
        self.grid.set(*hidden_obj.cur_pos, None)
        hidden_obj.cur_pos = None
        hidden_obj.init_pos = None
        box.contains = hidden_obj

        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(hidden_obj.type, hidden_obj.color)
        self.instrs = PickupInstr(desc)

class Level_GoToObjInBoxLocalR3(Level_ActionObjDoorR3):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        box = Box(self._rand_color())
        self.place_in_room(1, 0, box)

        # Add distractors and put one in the above box
        #   the box and the hidden object together are considered as one distractor
        #   since only one of them will be visible at a time
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors)
        hidden_obj = self._rand_elem(objs)
        self.grid.set(*hidden_obj.cur_pos, None)
        hidden_obj.cur_pos = None
        hidden_obj.init_pos = None
        box.contains = hidden_obj

        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(hidden_obj.type, hidden_obj.color)
        self.instrs = GoToInstr(desc)

class Level_GoToNeighborRoomR3(Level_ActionObjDoorR3):
    '''
    Doors are open.
    '''
    def __init__(self, seed=None):
        self.is_open = True
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        target_room_i = self._rand_elem([0, 2])
        target_room_objs = self.add_distractors(i=target_room_i, j=0, num_distractors=self.num_distractors)
        target_obj = self._rand_elem(target_room_objs)

        # Do not put the target object and boxes in the starting room
        # This level aims to teach the agent to explore a new room when the starting room is fully explored
        # The exploration of boxes in the starting room is excluded to control the complexity of the level
        exclude_objs = [(target_obj.type, target_obj.color)]
        for color in COLOR_NAMES:
            exclude_objs.append(('box', color))
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors, exclude_objs=exclude_objs)
        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked, is_open=self.is_open)
            objs.append(door)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(target_obj.type, target_obj.color)
        self.instrs = GoToInstr(desc)

class Level_PutNextLocalR3(Level_ActionObjDoorR3):
    '''
    Doors are closed.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors)
        two_objs = self._rand_subset(objs, 2)
        obj_to_move, obj_fixed = two_objs

        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)
            objs.append(door)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        # Randomly flip the object to be moved
        if self._rand_bool():
            t = obj_to_move
            obj_to_move = obj_fixed
            obj_fixed = t

        self.instrs = PutNextInstr(
            ObjDesc(obj_to_move.type, obj_to_move.color),
            ObjDesc(obj_fixed.type, obj_fixed.color)
        )

class Level_OpenBoxPickupLocal2BoxesR3(Level_ActionObjDoorR3):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):


        for door_color in COLOR_NAMES[:2]:
            door, _ = self.add_door(i=1, j=0, color=door_color, locked=False, is_open=False)

        boxes = []
        for box_color in COLOR_NAMES[:2]:
            box, pos = self.add_object(i=1, j=0, kind="box", color=box_color)
            boxes.append(box)

        target_box = self._rand_elem(boxes)

        objs = []
        for color in COLOR_NAMES[4:]:
            ball, pos = self.add_object(i=1, j=0, kind="ball", color=color)
            objs.append(ball)
            key, pos = self.add_object(i=1, j=0, kind="key", color=color)
            objs.append(key)
        hidden_obj = self._rand_elem(objs)
        self.grid.set(*hidden_obj.cur_pos, None)
        hidden_obj.cur_pos = None
        hidden_obj.init_pos = None
        target_box.contains = hidden_obj

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(hidden_obj.type, hidden_obj.color)
        self.instrs = PickupInstr(desc)

class Level_PutNextLocalBallBoxOneR3(Level_ActionObjDoorR3):
    '''
    The obj to pick and obj to drop_next_to are in the same room where the agent starts the mission
    Doors are open.
    There four balls in the agent's starting room. Each ball has different color.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        objs = []
        for ball_color in COLOR_NAMES[2:]:
            ball, pos = self.add_object(i=1, j=0, kind="ball", color=ball_color)
            objs.append(ball)
        two_objs = self._rand_subset(objs, 2)
        obj_to_move, obj_fixed = two_objs

        for door_color in COLOR_NAMES[:2]:
            door, _ = self.add_door(i=1, j=0, color=door_color, locked=False, is_open=False)

        for box_color in COLOR_NAMES[:2]:
            box, pos = self.add_object(i=1, j=0, kind="box", color=box_color)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        # Randomly flip the object to be moved
        if self._rand_bool():
            t = obj_to_move
            obj_to_move = obj_fixed
            obj_fixed = t

        self.instrs = PutNextInstr(
            ObjDesc(obj_to_move.type, obj_to_move.color),
            ObjDesc(obj_fixed.type, obj_fixed.color)
        )

class Level_PutNextLocalBallBoxTwoR3(Level_ActionObjDoorR3):
    '''
    The obj to pick up is inside a box
    The obj to put_next_to is not inside a box
    There four balls in the agent's starting room. Each ball has different color and one may hide in a box.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        boxes = []
        for box_color in COLOR_NAMES[:2]:
            box, pos = self.add_object(i=1, j=0, kind="box", color=box_color)
            boxes.append(box)
        target_box = self._rand_elem(boxes)

        objs = [] # balls
        for ball_color in COLOR_NAMES[2:]:
            ball, pos = self.add_object(i=1, j=0, kind="ball", color=ball_color)
            objs.append(ball)
        two_objs = self._rand_subset(objs, 2)
        obj_to_move, obj_fixed = two_objs

        # Randomly flip the object to be moved
        if self._rand_bool():
            t = obj_to_move
            obj_to_move = obj_fixed
            obj_fixed = t

        hidden_obj = obj_to_move
        self.grid.set(*hidden_obj.cur_pos, None)
        hidden_obj.cur_pos = None
        hidden_obj.init_pos = None
        target_box.contains = hidden_obj

        for door_color in COLOR_NAMES[:2]:
            door, _ = self.add_door(i=1, j=0, color=door_color, locked=False, is_open=False)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = PutNextInstr(
            ObjDesc(obj_to_move.type, obj_to_move.color),
            ObjDesc(obj_fixed.type, obj_fixed.color)
        )


class Level_PutNextLocalBallBoxThreeR3(Level_ActionObjDoorR3):
    '''
    The obj to pick up is not inside a box
    The obj to put_next_to is inside a box
    There four balls in the agent's starting room. Each ball has different color and one may hide in a box.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        boxes = []
        for box_color in COLOR_NAMES[:2]:
            box, pos = self.add_object(i=1, j=0, kind="box", color=box_color)
            boxes.append(box)
        target_box = self._rand_elem(boxes)

        objs = [] # balls
        for ball_color in COLOR_NAMES[2:]:
            ball, pos = self.add_object(i=1, j=0, kind="ball", color=ball_color)
            objs.append(ball)
        two_objs = self._rand_subset(objs, 2)
        obj_to_move, obj_fixed = two_objs

        # Randomly flip the object to be moved
        if self._rand_bool():
            t = obj_to_move
            obj_to_move = obj_fixed
            obj_fixed = t

        hidden_obj = obj_fixed
        self.grid.set(*hidden_obj.cur_pos, None)
        hidden_obj.cur_pos = None
        hidden_obj.init_pos = None
        target_box.contains = hidden_obj

        for door_color in COLOR_NAMES[:2]:
            door, _ = self.add_door(i=1, j=0, color=door_color, locked=False, is_open=False)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = PutNextInstr(
            ObjDesc(obj_to_move.type, obj_to_move.color),
            ObjDesc(obj_fixed.type, obj_fixed.color)
        )

class Level_PutNextLocalBallBoxFourR3(Level_ActionObjDoorR3):
    '''
    The obj to pick up is inside a box
    The obj to put_next_to is inside a box
    There four balls in the agent's starting room. Each ball has different color and two may hide in a box.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        boxes = []
        for box_color in COLOR_NAMES[:2]:
            box, pos = self.add_object(i=1, j=0, kind="box", color=box_color)
            boxes.append(box)

        objs = [] # balls
        for ball_color in COLOR_NAMES[2:]:
            ball, pos = self.add_object(i=1, j=0, kind="ball", color=ball_color)
            objs.append(ball)
        two_objs = self._rand_subset(objs, 2)
        obj_to_move, obj_fixed = two_objs

        # Randomly flip the object to be moved
        if self._rand_bool():
            t = obj_to_move
            obj_to_move = obj_fixed
            obj_fixed = t

        hidden_obj1 = obj_fixed
        self.grid.set(*hidden_obj1.cur_pos, None)
        hidden_obj1.cur_pos = None
        hidden_obj1.init_pos = None
        boxes[0].contains = hidden_obj1

        hidden_obj2 = obj_to_move
        self.grid.set(*hidden_obj2.cur_pos, None)
        hidden_obj2.cur_pos = None
        hidden_obj2.init_pos = None
        boxes[1].contains = hidden_obj2

        for door_color in COLOR_NAMES[:2]:
            door, _ = self.add_door(i=1, j=0, color=door_color, locked=False, is_open=False)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = PutNextInstr(
            ObjDesc(obj_to_move.type, obj_to_move.color),
            ObjDesc(obj_fixed.type, obj_fixed.color)
        )

'''
The obj to pick and obj to drop_next_to are in the same room where the agent starts the mission
    zero DoorPass
The obj to pick is in the agent's starting room and obj to drop_next_to is in another room.
    one DoorPass
The obj to pick is in another room and obj to drop_next_to is in the agent's starting room.
    two DoorPass
The obj to pick and obj to drop_next_to are in the same room which is not the agent's starting room.
    One DoorPass
The obj to pick and obj to drop_next_to scatter in the two different rooms, either of which is not the agent's starting room.
    Three DoorPass
'''

### Three-Subgoal Task Group
class Level_OpenGoToBallR3(Level_ActionObjDoorR3):
    '''
    Doors are closed.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        target_room_i = self._rand_elem([0, 2])
        target_ball_color = self._rand_elem(COLOR_NAMES)
        target_ball, pos = self.add_object(i=target_room_i, j=0, kind='ball', color=target_ball_color)

        target_room_objs = self.add_distractors(i=target_room_i, j=0, num_distractors=self.num_distractors)
        target_obj = target_ball

        # Do not put the target object and boxes in the starting room
        # This level aims to teach the agent to explore a new room when the starting room is fully explored
        # The exploration of boxes in the starting room is excluded to control the complexity of the level
        exclude_objs = [(target_obj.type, target_obj.color)]
        for color in COLOR_NAMES:
            exclude_objs.append(('box', color))
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors, exclude_objs=exclude_objs)
        for _ in range(self.num_doors):
            door_color=self._rand_elem(COLOR_NAMES[:3])
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked, color=door_color)
            objs.append(door)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(target_obj.type, target_obj.color)
        self.instrs = GoToInstr(desc)

class Level_OpenGoToR3(Level_ActionObjDoorR3):
    '''
    Doors are closed.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        target_room_i = self._rand_elem([0, 2])
        target_room_objs = self.add_distractors(i=target_room_i, j=0, num_distractors=self.num_distractors)
        target_obj = self._rand_elem(target_room_objs)

        # Do not put the target object and boxes in the starting room
        # This level aims to teach the agent to explore a new room when the starting room is fully explored
        # The exploration of boxes in the starting room is excluded to control the complexity of the level
        exclude_objs = [(target_obj.type, target_obj.color)]
        for color in COLOR_NAMES:
            exclude_objs.append(('box', color))
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors, exclude_objs=exclude_objs)
        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)
            objs.append(door)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(target_obj.type, target_obj.color)
        self.instrs = GoToInstr(desc)

class Level_UnlockGoToR3(Level_ActionObjDoorR3):
    '''
    Doors are locked.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed, door_locked=True)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)

        target_room_i = self._rand_elem([0, 2])
        target_room_objs = self.add_distractors(i=target_room_i, j=0, num_distractors=self.num_distractors)
        target_obj = self._rand_elem(target_room_objs)
        target_room = self.get_room(target_room_i, 0)
        # The following approach for retrieving the target door is excludsive to this level
        target_door_idx = target_room_i # respective to the target room not the starting room
        target_door = target_room.doors[target_door_idx]

        key, pos = self.add_object(i=1, j=0, kind='key', color=target_door.color)

        # Do not put the target object and boxes in the starting room
        # This level aims to teach the agent to explore a new room when the starting room is fully explored
        # The exploration of boxes in the starting room is excluded to control the complexity of the level
        exclude_objs = [(target_obj.type, target_obj.color)]
        for color in COLOR_NAMES:
            exclude_objs.append(('box', color))
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors-1, exclude_objs=exclude_objs)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(target_obj.type, target_obj.color)
        self.instrs = GoToInstr(desc)

class Level_UnblockPickupR3(Level_ActionObjDoorR3):
    # The doos is closed but not locked
    # The door is blocked by a key, box or ball

    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        # doors are open
        for color in COLOR_NAMES[:2]:
            door, _ = self.add_door(i=1, j=0, is_open=True, locked=self.door_locked, color=color)

        target_room_i = self._rand_elem([0, 2])
        target_room = self.get_room(target_room_i, 0)
        # The following approach for retrieving the target door is excludsive to this level
        target_door_idx = target_room_i # respective to the target room not the starting room
        blocked_obj = target_room.doors[target_door_idx]
        blocked_obj_i, blocked_obj_j = blocked_obj.cur_pos

        # add two keys, two boxes and two balls
        key_colors = COLOR_NAMES[:2]
        box_colors = COLOR_NAMES[2:4]
        ball_colors = COLOR_NAMES[4:]
        colors = {"key":key_colors, "box":box_colors, "ball":ball_colors}
        objs_typecolor = []
        for obj_type in ["ball", "box", "key"]:
            for obj_color in colors[obj_type]:
                objs_typecolor.append((obj_type, obj_color))
        objs_start_room = self._rand_subset(objs_typecolor, 3)
        objs_another_room = [obj for obj in objs_typecolor if obj not in objs_start_room]

        # Place the blocker
        blocker = self._rand_elem(objs_start_room)
        objs_start_room_wo_blocker = [obj for obj in objs_start_room if obj != blocker]
        blocker = WorldObj.decode(OBJECT_TO_IDX[blocker[0]], COLOR_TO_IDX[blocker[1]], 0)
        if target_room_i == 2: # the door connects to the room on the right
            blocker_i = blocked_obj_i - 1
        else: # target_room_i == 0, the door connects to the room on the left
            blocker_i = blocked_obj_i + 1
        blocker_pos = (blocker_i, blocked_obj_j)
        self.grid.set(*blocker_pos, blocker)
        blocker.init_pos = blocker_pos
        blocker.cur_pos = blocker_pos

        # Add the rest objects
        for obj in objs_start_room_wo_blocker:
            _, _ = self.add_object(i=1, j=0, kind=obj[0], color=obj[1])
        for obj in objs_another_room:
            _, _ = self.add_object(i=target_room_i, j=0, kind=obj[0], color=obj[1])

        # obj to pick up
        obj_to_pickup = self._rand_elem(objs_another_room)

        self.place_agent(i=1, j=0)

        desc = ObjDesc(obj_to_pickup[0], obj_to_pickup[1])
        self.instrs = PickupInstr(desc)

class Level_UnblockGoToDoorLocalR3(Level_ActionObjDoorR3):
    # The doos is closed but not locked
    # The door is blocked by a key, box or ball

    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)

        target_room_i = self._rand_elem([0, 2])
        target_room = self.get_room(target_room_i, 0)
        # The following approach for retrieving the target door is excludsive to this level
        target_door_idx = target_room_i # respective to the target room not the starting room
        target_obj = target_room.doors[target_door_idx]
        target_i, target_j = target_obj.cur_pos

        # Place the blocker
        kind = self._rand_elem(['key', 'ball', 'box'])
        color = self._rand_elem(COLOR_NAMES)
        blocker = WorldObj.decode(OBJECT_TO_IDX[kind], COLOR_TO_IDX[color], 0)
        if target_room_i == 2: # the door connects to the room on the right
            blocker_i = target_i - 1
        else: # target_room_i == 0, the door connects to the room on the left
            blocker_i = target_i + 1
        blocker_pos = (blocker_i, target_j)
        self.grid.set(*blocker_pos, blocker)
        blocker.init_pos = blocker_pos
        blocker.cur_pos = blocker_pos

        # Add the rest distracting objects in the starting room
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors-1)

        self.place_agent(i=1, j=0)

        desc = ObjDesc(target_obj.type, target_obj.color)
        self.instrs = GoToInstr(desc)

### Four-Subgoal Task Group
class Level_OpenPickupR3(Level_ActionObjDoorR3):
    '''
    Doors are closed.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        target_room_i = self._rand_elem([0, 2])
        target_room_objs = self.add_distractors(i=target_room_i, j=0, num_distractors=self.num_distractors)
        target_obj = self._rand_elem(target_room_objs)

        # Do not put the target object and boxes in the starting room
        # This level aims to teach the agent to explore a new room when the starting room is fully explored
        # The exploration of boxes in the starting room is excluded to control the complexity of the level
        exclude_objs = [(target_obj.type, target_obj.color)]
        for color in COLOR_NAMES:
            exclude_objs.append(('box', color))
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors, exclude_objs=exclude_objs)
        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)
            objs.append(door)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(target_obj.type, target_obj.color)
        self.instrs = PickupInstr(desc)

class Level_UnlockPickupR3(Level_ActionObjDoorR3):
    '''
    Doors are locked.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed, door_locked=True)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked)

        target_room_i = self._rand_elem([0, 2])
        target_room_objs = self.add_distractors(i=target_room_i, j=0, num_distractors=self.num_distractors)
        target_obj = self._rand_elem(target_room_objs)
        target_room = self.get_room(target_room_i, 0)
        # The following approach for retrieving the target door is excludsive to this level
        target_door_idx = target_room_i # respective to the target room not the starting room
        target_door = target_room.doors[target_door_idx]

        key, pos = self.add_object(i=1, j=0, kind='key', color=target_door.color)

        # Do not put the target object and boxes in the starting room
        # This level aims to teach the agent to explore a new room when the starting room is fully explored
        # The exploration of boxes in the starting room is excluded to control the complexity of the level
        exclude_objs = [(target_obj.type, target_obj.color)]
        for color in COLOR_NAMES:
            exclude_objs.append(('box', color))
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors-1, exclude_objs=exclude_objs)

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(target_obj.type, target_obj.color)
        self.instrs = PickupInstr(desc)

class Level_UnblockDoorGoToR3(Level_ActionObjDoorR3):
    # The doos is open

    def __init__(self, seed=None):
        self.is_open = True
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        for _ in range(self.num_doors):
            door, _ = self.add_door(i=1, j=0, locked=self.door_locked, is_open=self.is_open)

        target_room_i = self._rand_elem([0, 2])
        target_room_objs = self.add_distractors(i=target_room_i, j=0, num_distractors=self.num_distractors)
        target_obj = self._rand_elem(target_room_objs)
        target_room = self.get_room(target_room_i, 0)
        # The following approach for retrieving the target door is excludsive to this level
        target_door_idx = target_room_i # respective to the target room not the starting room
        target_door = target_room.doors[target_door_idx]
        target_i, target_j = target_door.cur_pos

        # Add the rest distracting objects in the starting room
        exclude_objs = [(target_obj.type, target_obj.color)]
        objs = self.add_distractors(i=1, j=0, num_distractors=self.num_distractors, exclude_objs=exclude_objs)

        # Move one object to block the target door
        blocker = self._rand_elem(objs)
        self.grid.set(*blocker.cur_pos, None)
        if target_room_i == 2: # the door connects to the room on the right
            blocker_i = target_i - 1
        else: # target_room_i == 0, the door connects to the room on the left
            blocker_i = target_i + 1
        blocker_pos = (blocker_i, target_j)
        blocker.init_pos = blocker_pos
        blocker.cur_pos = blocker_pos
        self.grid.set(*blocker_pos, blocker)

        self.place_agent(i=1, j=0)

        desc = ObjDesc(target_obj.type, target_obj.color)
        self.instrs = GoToInstr(desc)

# Levels that have 2 rooms
# Environments For Low-level Tasks
class Level_ActionObjDoorR2(RoomGridLevel):
    """
    [pick up an object] or
    [go to an object or door] or
    [open a door]
    (in the current room)
    """

    def __init__(self, seed=None, door_locked=False, num_distractors=5):
        self.door_locked = door_locked
        self.num_distractors = num_distractors
        self.num_doors = 1
        super().__init__(
            room_size=8,
            num_rows=1,
            num_cols=2,
            agent_view_size=7,
            seed=seed
        )

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors)
        door, _ = self.add_door(i=0, j=0, locked=self.door_locked)
        objs.append(door)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        obj = self._rand_elem(objs)
        desc = ObjDesc(obj.type, obj.color)

        if obj.type == 'door':
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = OpenInstr(desc)
        else:
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = PickupInstr(desc)


class Level_GoToLocalR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors)
        door, _ = self.add_door(i=0, j=0, locked=self.door_locked)
        objs.append(door)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        obj = self._rand_elem(objs)
        desc = ObjDesc(obj.type, obj.color)
        self.instrs = GoToInstr(desc)

class Level_OpenBoxLocalR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        # add a box
        color = self._rand_elem(COLOR_NAMES)
        type = 'box'
        obj = (type, color)
        box, pos = self.add_object(0, 0, *obj)

        # add distractors
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors-1)
        door, _ = self.add_door(i=0, j=0, locked=self.door_locked)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(box.type, box.color)
        self.instrs = OpenBoxInstr(desc)

class Level_OpenDoorLocalR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors)
        door, _ = self.add_door(i=0, j=0, locked=self.door_locked)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(door.type, door.color)
        self.instrs = OpenInstr(desc)

class Level_PassDoorLocalR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        self.door_locked = False
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors)
        door, _ = self.add_door(i=0, j=0, is_open=True, locked=self.door_locked)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(door.type, door.color)
        self.instrs = PassInstr(desc)

class Level_PickupLocalR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors)
        door, _ = self.add_door(i=0, j=0, locked=self.door_locked)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        obj = self._rand_elem(objs)
        desc = ObjDesc(obj.type, obj.color)
        self.instrs = PickupInstr(desc)

class Level_DropNextToLocalR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors)
        door, _ = self.add_door(i=0, j=0, locked=self.door_locked)
        objs.append(door)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        next_to_obj = self._rand_elem(objs)

        carried_obj_color = self._rand_elem(COLOR_NAMES)
        carried_obj_type = self._rand_elem(['key', 'ball', 'box'])

        self.carrying = WorldObj.decode(OBJECT_TO_IDX[carried_obj_type], COLOR_TO_IDX[carried_obj_color], 0)
        self.carrying.cur_pos = np.array([-1, -1])

        self.instrs = DropNextInstr(
            obj_carried = ObjDesc(carried_obj_type, carried_obj_color),
            obj_fixed = ObjDesc(next_to_obj.type, next_to_obj.color),
            initially_carried_world_obj = self.carrying
        )

class Level_DropNotNextToLocalR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors)
        door, _ = self.add_door(i=0, j=0, locked=self.door_locked)
        objs.append(door)

        self.place_agent(i=0, j=0)
        
        # Make sure no unblocking is required
        self.check_objs_reachable()

        not_next_to_obj = self._rand_elem(objs)

        carried_obj_color = self._rand_elem(COLOR_NAMES)
        carried_obj_type = self._rand_elem(['key', 'ball', 'box'])

        self.carrying = WorldObj.decode(OBJECT_TO_IDX[carried_obj_type], COLOR_TO_IDX[carried_obj_color], 0)
        self.carrying.cur_pos = np.array([-1, -1])

        self.instrs = DropNotNextInstr(
            obj_carried = ObjDesc(carried_obj_type, carried_obj_color),
            obj_fixed = ObjDesc(not_next_to_obj.type, not_next_to_obj.color),
            initially_carried_world_obj = self.carrying
        )

class Level_DropNextNothingLocalR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors)
        door, _ = self.add_door(i=0, j=0, locked=self.door_locked)
        objs.append(door)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        carried_obj_color = self._rand_elem(COLOR_NAMES)
        carried_obj_type = self._rand_elem(['key', 'ball', 'box'])

        self.carrying = WorldObj.decode(OBJECT_TO_IDX[carried_obj_type], COLOR_TO_IDX[carried_obj_color], 0)

        self.instrs = DropNextNothingInstr(
            initially_carried_world_obj = self.carrying,
            obj_to_drop = ObjDesc(carried_obj_type, carried_obj_color)
        )

### Envs for paper
'''
    The env has 1 cloded door of any color, 1 blue box, red and green balls.
    The target ball, either red or green, is hidden in the box.
    The goal is to find and pickup the target ball.
    Set of 3 subgoals:
        Open the blue box
        Pickup the red ball
        Pickup the green ball
'''
class Level_DiscoverHiddenBallBlueBoxR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        # add a box
        color = 'blue'
        type = 'box'
        obj = (type, color)
        target_box, pos = self.add_object(0, 0, *obj)

        # add a closed door
        door, _ = self.add_door(i=0, j=0, locked=False, is_open=False)

        # add two balls
        balls = []
        for color in ['red', 'green']:
            for type in ['ball']:
                obj = (type, color)
                obj, pos = self.add_object(0, 0, *obj)
                balls.append(obj)

        hidden_obj = self._rand_elem(balls)
        self.grid.set(*hidden_obj.cur_pos, None)
        hidden_obj.cur_pos = None
        hidden_obj.init_pos = None
        target_box.contains = hidden_obj

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(hidden_obj.type, hidden_obj.color)
        self.instrs = PickupInstr(desc)

'''
    The env has 1 cloded door of any color, blue and purple boxes, red and green balls keys.
    The target key, either yellow or grey, is hidden in the target box, either blue or purple.
    The goal is to find and pickup the target key.
    Set of 4 subgoals:
        Open the blue box
        Open the purple box
        Pickup the red key
        Pickup the green key
'''
class Level_DiscoverHiddenKeyR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        # add a closed door
        door, _ = self.add_door(i=0, j=0, locked=False, is_open=False)

        # add boxes
        type = 'box'
        boxes = []
        for box_color in ['blue', 'purple']:
            box = (type, box_color)
            box, pos = self.add_object(0, 0, *box)
            boxes.append(box)
        target_box = self._rand_elem(boxes)

        # add two balls and two keys
        keys = []
        for color in ['red', 'green']:
            for type in ['ball', 'key']:
                obj = (type, color)
                obj, pos = self.add_object(0, 0, *obj)
                if type == 'key':
                    keys.append(obj)
        hidden_obj = self._rand_elem(keys)

        # hide the key in the box
        self.grid.set(*hidden_obj.cur_pos, None)
        hidden_obj.cur_pos = None
        hidden_obj.init_pos = None
        target_box.contains = hidden_obj

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(hidden_obj.type, hidden_obj.color)
        self.instrs = PickupInstr(desc)

'''
The env has 1 door, 2 keys, 1 ball, 1 box.
The box, the ball and one key are distractors.
Each door or each key may have a color of 'red' or 'green'.
The ball or the box may have a random color from COLOR_NAMES.
Subgoals:
    Open the red door
    Open the green door
    Pickup the red key
    Pickup the green key
'''
class Level_UnlockLocalSmallR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed, door_locked=True)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        door_color = self._rand_elem(['red', 'green'])
        target_door, _ = self.add_door(i=0, j=0, color=door_color, locked=True, is_open=False)

        for key_color in ['red', 'green']:
            key, _ = self.add_object(0, 0, 'key', key_color)
        for obj_type in ['ball', 'box']:
            obj_color = self._rand_elem(COLOR_NAMES)
            obj, _ = self.add_object(0, 0, obj_type, obj_color)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(target_door.type, target_door.color)
        self.instrs = OpenInstr(desc)


'''
The env has 1 door, 4 balls.
The door is clsoded and may have a color of 'red' or 'green'.
The balls may have a random color from ['red', 'green', 'blue', 'purple'].
One of the 4 balls is used to blcok the door.
Subgoals:
    GoTo the red door
    GoTo the green door
    Pickup the red ball
    Pickup the green ball
    Pickup the blue ball
    Pickup the purple ball
'''
class Level_UnblockGoToDoorR2(Level_ActionObjDoorR2):
    # The doos is closed but not locked
    # The door is blocked by a key, box or ball

    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        door_color = self._rand_elem(['red', 'green'])
        target_door, _ = self.add_door(i=0, j=0, color=door_color, locked=False, is_open=False)
        target_i, target_j = target_door.cur_pos

        # Place the blocker
        blocker_type = 'ball'
        blocker_color = self._rand_elem(['red', 'green', 'blue', 'purple'])
        blocker = WorldObj.decode(OBJECT_TO_IDX[blocker_type], COLOR_TO_IDX[blocker_color], 0)
        blocker_i = target_i - 1 # the blocker should be on the left of the target door
        blocker_pos = (blocker_i, target_j)
        self.grid.set(*blocker_pos, blocker)
        blocker.init_pos = blocker_pos
        blocker.cur_pos = blocker_pos

        # Add the rest distracting objects (balls) in the starting room
        for color in ['red', 'green', 'blue', 'purple']:
            if color!=blocker_color:
                obj, _ = self.add_object(0, 0, 'ball', color)

        self.place_agent(i=0, j=0)

        desc = ObjDesc(target_door.type, target_door.color)
        self.instrs = GoToInstr(desc)

'''
The env has 1 door, 2 balls, 2 keys and 1 box.
Balls and keys are possible objects to arrange.
Balls: red ball and green ball
Keys: red key and green key
Box: a random color from COLOR_NAMES
Door: a random color from COLOR_NAMES, closed
Subgoals:
    Pickup the red ball
    Pickup the green ball
    Pickup the red key
    Pickup the green key
    Drop next to the red ball
    Drop next to the green ball
    Drop next to the red key
    Drop next to the green key
Mission: move one of the balls or keys to one of the balls or keys
'''
class Level_Arrangement1R2(Level_ActionObjDoorR2):
    '''
    Doors are closed.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        door_color = self._rand_elem(COLOR_NAMES)
        door, _ = self.add_door(i=0, j=0, color=door_color, locked=False, is_open=False)

        box_color = self._rand_elem(COLOR_NAMES)
        box, _ = self.add_object(0, 0, 'box', box_color)

        objs = []
        for obj_type in ['ball', 'key']:
            for color in ['red', 'green']:
                obj, _ = self.add_object(0, 0, obj_type, color)
                objs.append(obj)
        two_objs = self._rand_subset(objs, 2)
        obj_to_move, obj_fixed = two_objs

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        # Randomly flip the object to be moved
        if self._rand_bool():
            t = obj_to_move
            obj_to_move = obj_fixed
            obj_fixed = t

        self.instrs = PutNextInstr(
            ObjDesc(obj_to_move.type, obj_to_move.color),
            ObjDesc(obj_fixed.type, obj_fixed.color)
        )

'''
Randomly select three objects from set of 2 balls and 2 keys.
Move the first object to the second object.
Move the second object to the third object.
'''
class Level_Arrangement2R2(Level_ActionObjDoorR2):
    '''
    Doors are closed.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        door_color = self._rand_elem(COLOR_NAMES)
        door, _ = self.add_door(i=0, j=0, color=door_color, locked=False, is_open=False)

        box_color = self._rand_elem(COLOR_NAMES)
        box, _ = self.add_object(0, 0, 'box', box_color)

        objs = []
        for obj_type in ['ball', 'key']:
            for color in ['red', 'green']:
                obj, _ = self.add_object(0, 0, obj_type, color)
                objs.append(obj)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        three_objs = self._rand_subset(objs, 3)
        first_obj, second_obj, third_obj = three_objs

        # Randomly flip the first and second objects
        if self._rand_bool():
            t = first_obj
            first_obj = second_obj
            second_obj = t

        # Move the first object to the second object
        first_instr = PutNextInstr(
            ObjDesc(first_obj.type, first_obj.color),
            ObjDesc(second_obj.type, second_obj.color)
        )
        # Move the second object to the third object
        second_instr = PutNextInstr(
            ObjDesc(second_obj.type, second_obj.color),
            ObjDesc(third_obj.type, third_obj.color)
        )

        self.instrs = AndInstr(first_instr, second_instr)


## High-level tasks
### Two-Subgoal Task Group
class Level_UnlockLocalR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed, door_locked=True)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        target_door, _ = self.add_door(i=0, j=0, locked=self.door_locked)

        key, _ = self.add_object(0, 0, 'key', target_door.color)
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors-1)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(target_door.type, target_door.color)
        self.instrs = OpenInstr(desc)

class Level_PickupObjInBoxLocalR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        box = Box(self._rand_color())
        self.place_in_room(0, 0, box)

        # Add distractors and put one in the above box
        #   the box and the hidden object together are considered as one distractor
        #   since only one of them will be visible at a time
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors)
        hidden_obj = self._rand_elem(objs)
        self.grid.set(*hidden_obj.cur_pos, None)
        hidden_obj.cur_pos = None
        hidden_obj.init_pos = None
        box.contains = hidden_obj

        door, _ = self.add_door(i=0, j=0, locked=self.door_locked)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(hidden_obj.type, hidden_obj.color)
        self.instrs = PickupInstr(desc)

class Level_GoToObjInBoxLocalR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        box = Box(self._rand_color())
        self.place_in_room(0, 0, box)

        # Add distractors and put one in the above box
        #   the box and the hidden object together are considered as one distractor
        #   since only one of them will be visible at a time
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors)
        hidden_obj = self._rand_elem(objs)
        self.grid.set(*hidden_obj.cur_pos, None)
        hidden_obj.cur_pos = None
        hidden_obj.init_pos = None
        box.contains = hidden_obj

        door, _ = self.add_door(i=0, j=0, locked=self.door_locked)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(hidden_obj.type, hidden_obj.color)
        self.instrs = GoToInstr(desc)

class Level_GoToNeighborRoomR2(Level_ActionObjDoorR2):
    '''
    Doors are open.
    '''
    def __init__(self, seed=None):
        self.is_open = True
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        target_room_i = 1
        target_room_objs = self.add_distractors(i=target_room_i, j=0, num_distractors=self.num_distractors)
        target_obj = self._rand_elem(target_room_objs)

        # Do not put the target object and boxes in the starting room
        # This level aims to teach the agent to explore a new room when the starting room is fully explored
        exclude_objs = [(target_obj.type, target_obj.color)]
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors, exclude_objs=exclude_objs)
        door, _ = self.add_door(i=0, j=0, locked=self.door_locked, is_open=self.is_open)
        objs.append(door)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(target_obj.type, target_obj.color)
        self.instrs = GoToInstr(desc)

class Level_PutNextLocalR2(Level_ActionObjDoorR2):
    '''
    Doors are closed.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors)
        two_objs = self._rand_subset(objs, 2)
        obj_to_move, obj_fixed = two_objs

        door, _ = self.add_door(i=0, j=0, locked=self.door_locked)
        objs.append(door)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        # Randomly flip the object to be moved
        if self._rand_bool():
            t = obj_to_move
            obj_to_move = obj_fixed
            obj_fixed = t

        self.instrs = PutNextInstr(
            ObjDesc(obj_to_move.type, obj_to_move.color),
            ObjDesc(obj_fixed.type, obj_fixed.color)
        )

class Level_OpenBoxPickupLocal2BoxesR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):


        door_color = self._rand_elem(COLOR_NAMES[:2])
        door, _ = self.add_door(i=1, j=0, color=door_color, locked=False, is_open=False)

        boxes = []
        for box_color in COLOR_NAMES[:2]:
            box, pos = self.add_object(i=1, j=0, kind="box", color=box_color)
            boxes.append(box)

        target_box = self._rand_elem(boxes)

        objs = []
        for color in COLOR_NAMES[4:]:
            ball, pos = self.add_object(i=1, j=0, kind="ball", color=color)
            objs.append(ball)
            key, pos = self.add_object(i=1, j=0, kind="key", color=color)
            objs.append(key)
        hidden_obj = self._rand_elem(objs)
        self.grid.set(*hidden_obj.cur_pos, None)
        hidden_obj.cur_pos = None
        hidden_obj.init_pos = None
        target_box.contains = hidden_obj

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(hidden_obj.type, hidden_obj.color)
        self.instrs = PickupInstr(desc)

class Level_OpenBoxPickupLocal2Boxes2BallsR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):


        door_color = self._rand_elem(COLOR_NAMES[:2])
        door, _ = self.add_door(i=1, j=0, color=door_color, locked=False, is_open=False)

        boxes = []
        for box_color in COLOR_NAMES[:2]:
            box, pos = self.add_object(i=1, j=0, kind="box", color=box_color)
            boxes.append(box)

        target_box = self._rand_elem(boxes)

        objs = []
        for color in COLOR_NAMES[4:]:
            ball, pos = self.add_object(i=1, j=0, kind="ball", color=color)
            objs.append(ball)

        hidden_obj = self._rand_elem(objs)
        self.grid.set(*hidden_obj.cur_pos, None)
        hidden_obj.cur_pos = None
        hidden_obj.init_pos = None
        target_box.contains = hidden_obj

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(hidden_obj.type, hidden_obj.color)
        self.instrs = PickupInstr(desc)

class Level_OpenBoxPickupLocal2Boxes1BallR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):


        door_color = self._rand_elem(COLOR_NAMES[:2])
        door, _ = self.add_door(i=1, j=0, color=door_color, locked=False, is_open=False)

        boxes = []
        for box_color in COLOR_NAMES[:2]:
            box, pos = self.add_object(i=1, j=0, kind="box", color=box_color)
            boxes.append(box)

        target_box = self._rand_elem(boxes)

        objs = []
        color = self._rand_elem(COLOR_NAMES[4:])
        ball, pos = self.add_object(i=1, j=0, kind="ball", color=color)
        objs.append(ball)

        hidden_obj = self._rand_elem(objs)
        self.grid.set(*hidden_obj.cur_pos, None)
        hidden_obj.cur_pos = None
        hidden_obj.init_pos = None
        target_box.contains = hidden_obj

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(hidden_obj.type, hidden_obj.color)
        self.instrs = PickupInstr(desc)

class Level_OpenBoxPickupLocal1Box1BallR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):


        door_color = self._rand_elem(COLOR_NAMES[:2])
        door, _ = self.add_door(i=1, j=0, color=door_color, locked=False, is_open=False)

        boxes = []
        box_color = self._rand_elem(COLOR_NAMES[:2])
        box, pos = self.add_object(i=1, j=0, kind="box", color=box_color)
        boxes.append(box)

        target_box = self._rand_elem(boxes)

        objs = []
        color = self._rand_elem(COLOR_NAMES[4:])
        ball, pos = self.add_object(i=1, j=0, kind="ball", color=color)
        objs.append(ball)

        hidden_obj = self._rand_elem(objs)
        self.grid.set(*hidden_obj.cur_pos, None)
        hidden_obj.cur_pos = None
        hidden_obj.init_pos = None
        target_box.contains = hidden_obj

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(hidden_obj.type, hidden_obj.color)
        self.instrs = PickupInstr(desc)

class Level_OpenBoxPickupLocal1Box1BallFixColorR2(Level_ActionObjDoorR2):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        door_color = COLOR_NAMES[0]
        door, _ = self.add_door(i=1, j=0, color=door_color, locked=False, is_open=False)

        box_color = COLOR_NAMES[1]
        box, pos = self.add_object(i=1, j=0, kind="box", color=box_color)

        ball_color = COLOR_NAMES[2]
        ball, pos = self.add_object(i=1, j=0, kind="ball", color=ball_color)

        hidden_obj = ball
        self.grid.set(*hidden_obj.cur_pos, None)
        hidden_obj.cur_pos = None
        hidden_obj.init_pos = None
        box.contains = hidden_obj

        self.place_agent(i=1, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(hidden_obj.type, hidden_obj.color)
        self.instrs = PickupInstr(desc)

### Three-Subgoal Task Group
class Level_OpenGoToR2(Level_ActionObjDoorR2):
    '''
    Doors are closed.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        target_room_i = 1
        target_room_objs = self.add_distractors(i=target_room_i, j=0, num_distractors=self.num_distractors)
        target_obj = self._rand_elem(target_room_objs)

        # Do not put the target object and boxes in the starting room
        # This level aims to teach the agent to explore a new room when the starting room is fully explored
        exclude_objs = [(target_obj.type, target_obj.color)]
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors, exclude_objs=exclude_objs)
        door, _ = self.add_door(i=0, j=0, locked=self.door_locked)
        objs.append(door)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(target_obj.type, target_obj.color)
        self.instrs = GoToInstr(desc)

class Level_UnlockGoToR2(Level_ActionObjDoorR2):
    '''
    Doors are locked.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed, door_locked=True)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):


        target_door, _ = self.add_door(i=0, j=0, locked=self.door_locked)
        target_room_i = 1
        target_room_objs = self.add_distractors(i=target_room_i, j=0, num_distractors=self.num_distractors)
        target_obj = self._rand_elem(target_room_objs)
        key, pos = self.add_object(i=0, j=0, kind='key', color=target_door.color)

        # Do not put the target object and boxes in the starting room
        # This level aims to teach the agent to explore a new room when the starting room is fully explored
        exclude_objs = [(target_obj.type, target_obj.color)]
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors-1, exclude_objs=exclude_objs)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(target_obj.type, target_obj.color)
        self.instrs = GoToInstr(desc)

class Level_UnblockGoToDoorLocalR2(Level_ActionObjDoorR2):
    # The doos is closed but not locked
    # The door is blocked by a key, box or ball

    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        target_door, _ = self.add_door(i=0, j=0, locked=self.door_locked)

        target_i, target_j = target_door.cur_pos

        # Place the blocker
        kind = self._rand_elem(['key', 'ball', 'box'])
        color = self._rand_elem(COLOR_NAMES)
        blocker = WorldObj.decode(OBJECT_TO_IDX[kind], COLOR_TO_IDX[color], 0)
        blocker_i = target_i - 1 # the blocker should be on the left of the target door
        blocker_pos = (blocker_i, target_j)
        self.grid.set(*blocker_pos, blocker)
        blocker.init_pos = blocker_pos
        blocker.cur_pos = blocker_pos

        # Add the rest distracting objects in the starting room
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors-1)

        self.place_agent(i=0, j=0)

        desc = ObjDesc(target_door.type, target_door.color)
        self.instrs = GoToInstr(desc)

### Four-Subgoal Task Group
class Level_OpenPickupR2(Level_ActionObjDoorR2):
    '''
    Doors are closed.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        target_room_i = 1
        target_room_objs = self.add_distractors(i=target_room_i, j=0, num_distractors=self.num_distractors)
        target_obj = self._rand_elem(target_room_objs)

        # Do not put the target object and boxes in the starting room
        # This level aims to teach the agent to explore a new room when the starting room is fully explored
        exclude_objs = [(target_obj.type, target_obj.color)]
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors, exclude_objs=exclude_objs)
        door, _ = self.add_door(i=0, j=0, locked=self.door_locked)
        objs.append(door)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(target_obj.type, target_obj.color)
        self.instrs = PickupInstr(desc)

class Level_UnlockPickupR2(Level_ActionObjDoorR2):
    '''
    Doors are locked.
    '''
    def __init__(self, seed=None):
        super().__init__(seed=seed, door_locked=True)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        target_door, _ = self.add_door(i=0, j=0, locked=self.door_locked)
        target_room_i = 1
        target_room_objs = self.add_distractors(i=target_room_i, j=0, num_distractors=self.num_distractors)
        target_obj = self._rand_elem(target_room_objs)
        key, pos = self.add_object(i=0, j=0, kind='key', color=target_door.color)

        # Do not put the target object and boxes in the starting room
        # This level aims to teach the agent to explore a new room when the starting room is fully explored
        exclude_objs = [(target_obj.type, target_obj.color)]
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors-1, exclude_objs=exclude_objs)

        self.place_agent(i=0, j=0)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        desc = ObjDesc(target_obj.type, target_obj.color)
        self.instrs = PickupInstr(desc)

class Level_UnblockDoorGoToR2(Level_ActionObjDoorR2):
    # The doos is open

    def __init__(self, seed=None):
        self.is_open = True
        super().__init__(seed=seed)

    # For member functions, add_distractors, add_door, place_agent,
    # their arguments i and j correspond to the column and row of the grid.
    def gen_mission(self):

        target_door, _ = self.add_door(i=0, j=0, locked=self.door_locked, is_open=self.is_open)
        target_room_i = 1
        target_room_objs = self.add_distractors(i=target_room_i, j=0, num_distractors=self.num_distractors)
        target_obj = self._rand_elem(target_room_objs)
        target_i, target_j = target_door.cur_pos

        # Add the rest distracting objects in the starting room
        exclude_objs = [(target_obj.type, target_obj.color)]
        objs = self.add_distractors(i=0, j=0, num_distractors=self.num_distractors, exclude_objs=exclude_objs)

        # Move one object to block the target door
        blocker = self._rand_elem(objs)
        self.grid.set(*blocker.cur_pos, None)
        blocker_i = target_i - 1 # the blocker is on the left of the target door
        blocker_pos = (blocker_i, target_j)
        blocker.init_pos = blocker_pos
        blocker.cur_pos = blocker_pos
        self.grid.set(*blocker_pos, blocker)

        self.place_agent(i=0, j=0)

        desc = ObjDesc(target_obj.type, target_obj.color)
        self.instrs = GoToInstr(desc)

for name, level in list(globals().items()):
    if name.startswith('Level_'):
        level.is_bonus = True

# Register the levels in this file
register_levels(__name__, globals())
