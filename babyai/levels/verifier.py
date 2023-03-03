import os
import numpy as np
from enum import Enum
from gym_minigrid.minigrid import WorldObj, COLOR_TO_IDX, OBJECT_TO_IDX, COLOR_NAMES, DIR_TO_VEC

# Object types we are allowed to describe in language
OBJ_TYPES = ['box', 'ball', 'key', 'door']

# Object types we are allowed to describe in language
OBJ_TYPES_NOT_DOOR = list(filter(lambda t: t != 'door', OBJ_TYPES))

# Locations are all relative to the agent's starting position
LOC_NAMES = ['left', 'right', 'front', 'behind']

SKILL_DESCRIPTIONS = ["GoTo", "OpenBox", "OpenDoor", "PassDoor", "Pickup", "DropNextTo", "DropNextNothing"]

# Environment flag to indicate that done actions should be
# used by the verifier
use_done_actions = os.environ.get('BABYAI_DONE_ACTIONS', False)


def dot_product(v1, v2):
    """
    Compute the dot product of the vectors v1 and v2.
    """

    return sum([i * j for i, j in zip(v1, v2)])


def pos_next_to(pos_a, pos_b):
    """
    Test if two positions are next to each other.
    The positions have to line up either horizontally or vertically,
    but positions that are diagonally adjacent are not counted.
    """

    xa, ya = pos_a
    xb, yb = pos_b
    d = abs(xa - xb) + abs(ya - yb)
    return d == 1


class LowlevelInstrSet:
    """
    The set of all low-level instructions and its valid version for a specific environment.
    Each instruction corresponds to one low-level subgoal.
    Each instruction is an interaction between the agent and one object.
    The object is an instance of ObjDesc.
    """

    def __init__(self, object_types=None, object_colors=None, subgoal_set_type=None):
        self.object_types = object_types or OBJ_TYPES
        self.object_colors = object_colors or COLOR_NAMES
        self.subgoal_set_type = subgoal_set_type or "subgoal_set_for_all"
        self.num_subgoals_info = {"total": 0}
        for skill_desc in SKILL_DESCRIPTIONS:
            self.num_subgoals_info[skill_desc] = 0

        self.all_subgoals = self.generate_all_subgoals(subgoal_set_type=subgoal_set_type)
        self.initial_valid_subgoals = []
        self.current_valid_subgoals = []

    def generate_instr_desc(self, instr, acticle='a'):
        assert instr.desc and instr.desc.type and instr.desc.color
        desc = instr.desc
        obj_desc = acticle + " " + desc.color + " " + desc.type

        if isinstance(instr, OpenInstr): # OpenDoorLocal
            instr.instr_desc = 'open ' + obj_desc
        elif isinstance(instr, PassInstr): # PassDoorLocal
            instr.instr_desc = 'pass ' + obj_desc
        elif isinstance(instr, OpenBoxInstr): # OpenBoxLocal
            instr.instr_desc = 'open ' + obj_desc
        elif isinstance(instr, GoToInstr): # GoToLocal
            instr.instr_desc = 'go to ' + obj_desc
        elif isinstance(instr, PickupInstr): # PickupLocal
            instr.instr_desc = 'pick up ' + obj_desc
        elif isinstance(instr, DropNextInstr): # DropNextToLocal
            instr.instr_desc = 'drop next to ' + obj_desc
        elif isinstance(instr, DropNextNothingInstr): # DropNextNothingLocal
            instr.instr_desc = 'drop ' + obj_desc
        else:
            err_msg = f"Instruction type ({type(instr)}) of a non-supported low-level task."
            raise TypeError(err_msg)

    def subgol_set_for_all(self):
        subgoal_instructions_by_skill = {}
        for skill_desc in SKILL_DESCRIPTIONS:
            subgoal_instructions_by_skill[skill_desc] = []

        for obj_type in self.object_types:
            for color in self.object_colors:
                obj = ObjDesc(obj_type, color=color)
                if obj_type == 'door':
                    subgoal_instructions_by_skill['OpenDoor'].append(OpenInstr(obj))
                    subgoal_instructions_by_skill['PassDoor'].append(PassInstr(obj))
                else:
                    subgoal_instructions_by_skill['DropNextNothing'].append(DropNextNothingInstr(initially_carried_world_obj=None, obj_to_drop=obj))
                    subgoal_instructions_by_skill['Pickup'].append(PickupInstr(obj))
                    if obj_type == 'box':
                        subgoal_instructions_by_skill['OpenBox'].append(OpenBoxInstr(obj))

                subgoal_instructions_by_skill['GoTo'].append(GoToInstr(obj))
                subgoal_instructions_by_skill['DropNextTo'].append(DropNextInstr(obj_carried=None, obj_fixed=obj, initially_carried_world_obj=None))
        return subgoal_instructions_by_skill

    def subgoal_set_for_PutNextLocalBallBox(self):
        subgoal_instructions_by_skill = {}
        for skill_desc in SKILL_DESCRIPTIONS:
            subgoal_instructions_by_skill[skill_desc] = []

        for obj_type in self.object_types:
            for color in self.object_colors:
                obj = ObjDesc(obj_type, color=color)
                if obj_type == "box" and color in COLOR_NAMES[:2]:
                    subgoal_instructions_by_skill['OpenBox'].append(OpenBoxInstr(obj))
                if obj_type == 'ball' and color in COLOR_NAMES[2:]:
                    subgoal_instructions_by_skill['Pickup'].append(PickupInstr(obj))
                    subgoal_instructions_by_skill['DropNextTo'].append(DropNextInstr(obj_carried=None, obj_fixed=obj, initially_carried_world_obj=None))
        return subgoal_instructions_by_skill

    def subgoal_set_for_OpenBoxPickupLocal2Boxes(self):
        subgoal_instructions_by_skill = {}
        for skill_desc in SKILL_DESCRIPTIONS:
            subgoal_instructions_by_skill[skill_desc] = []

        for obj_type in self.object_types:
            for color in self.object_colors:
                obj = ObjDesc(obj_type, color=color)
                if (obj_type == "key" or obj_type == "ball") and (color in COLOR_NAMES[4:]):
                    subgoal_instructions_by_skill['Pickup'].append(PickupInstr(obj))
                if obj_type == 'box' and color in COLOR_NAMES[:2]:
                    subgoal_instructions_by_skill['OpenBox'].append(OpenBoxInstr(obj))

        return subgoal_instructions_by_skill

    def subgol_set_for_UnblockPickup(self):
        subgoal_instructions_by_skill = {}
        for skill_desc in SKILL_DESCRIPTIONS:
            subgoal_instructions_by_skill[skill_desc] = []

        for obj_type in self.object_types:
            for color in self.object_colors:
                obj = ObjDesc(obj_type, color=color)
                if (obj_type == 'door') and (color in COLOR_NAMES[:2]):
                        subgoal_instructions_by_skill['PassDoor'].append(PassInstr(obj))
                elif (obj_type=="key" and color in COLOR_NAMES[:2]) or (obj_type == 'box' and color in COLOR_NAMES[2:4]) or (obj_type == "ball" and color in COLOR_NAMES[4:]):
                    subgoal_instructions_by_skill['DropNextNothing'].append(DropNextNothingInstr(initially_carried_world_obj=None, obj_to_drop=obj))
                    subgoal_instructions_by_skill['Pickup'].append(PickupInstr(obj))
                '''
                if (obj_type == 'door'):
                    if (color in COLOR_NAMES[4:]):
                        subgoal_instructions_by_skill['PassDoor'].append(PassInstr(obj))
                else:
                    if color in COLOR_NAMES[4:]:
                    subgoal_instructions_by_skill['DropNextNothing'].append(DropNextNothingInstr(initially_carried_world_obj=None, obj_to_drop=obj))
                    subgoal_instructions_by_skill['Pickup'].append(PickupInstr(obj))
                '''

        return subgoal_instructions_by_skill

    def fetch_subgoal_set(self, subgoal_set_type):
        if subgoal_set_type=="subgoal_set_for_all":
            return self.subgol_set_for_all()
        elif subgoal_set_type=="subgoal_set_for_PutNextLocalBallBox":
            return self.subgoal_set_for_PutNextLocalBallBox()
        elif subgoal_set_type=="subgoal_set_for_OpenBoxPickupLocal2Boxes":
            return self.subgoal_set_for_OpenBoxPickupLocal2Boxes()
        elif subgoal_set_type=="subgoal_set_for_UnblockPickupR3":
            return self.subgol_set_for_UnblockPickup()
        else:
            raise ValueError("Unknown subgoal set type: %s" % subgoal_set_type)

    def generate_all_subgoals(self, subgoal_set_type="subgoal_set_for_all"):
        """
        7 types of low-level instructions (subgoals)
        OpenDoorLocal:
            If the door is closed, open it; 
        PassDoorLocal:
            The door is open and unblocked on both sides.
        PickupLocal:
            If nothing has been carried, then an object, like a key, box, ball, can be picked.
        DropNextToLocal:
            The drop location is next to an object, key, box, ball.
            Can only drop onto an 'empty' cell.
        DropNextNothingLocal:
            There is no object (box, ball, key and door) nearby the drop location.
        OpenBoxLocal:
            open the box and uncover the object inside it if there is one.
            The opened box will be removed from the environment grid.
        GoToLocal:
            Only verify when the performed action is 'left', 'right', or 'forward'.

        No difference between 'a' and 'the' 
        """

        subgoal_instructions_by_skill = self.fetch_subgoal_set(subgoal_set_type)

        subgoals = []
        subgoal_idx = 0
        for skill_desc in SKILL_DESCRIPTIONS:
            self.num_subgoals_info[skill_desc] = len(subgoal_instructions_by_skill[skill_desc])
            for subgoal_instr in subgoal_instructions_by_skill[skill_desc]:
                self.generate_instr_desc(subgoal_instr, acticle='a')
                # subgoal: 0-subgoal_idx, 1-subgoal_instr, 2-skill_desc
                subgoals.append((subgoal_idx, subgoal_instr, skill_desc))
                subgoal_idx += 1
        # The variable subgoal_idx is the total number of subgoals
        self.num_subgoals_info['total'] = subgoal_idx

        return subgoals

    def display_all_subgoals(self):
        for subgoal in self.all_subgoals:
            print(f"[{subgoal[0]}] {subgoal[1].instr_desc}")

    def reset_valid_subgoals(self, env):
        self.initial_valid_subgoals = self.filter_valid_subgoals(self.all_subgoals, env, need_reset=True)
        self.current_valid_subgoals = self.initial_valid_subgoals.copy()

    # subgoals: [(subgoal_idx, subgoal_instr)]
    def filter_valid_subgoals(self, subgoals, env, need_reset=False):
        valid_subgoals = []
        for subgoal in subgoals:
            instr = subgoal[1]
            if need_reset:
                instr.reset_verifier(env)
            else:
                instr.desc.find_matching_objs(env, use_location=True)

            is_valid = False
            if isinstance(instr, DropNextNothingInstr):
                carried_expected_obj = env.carrying and (env.carrying.type==instr.desc.type and env.carrying.color==instr.desc.color)                
                obj_to_drop_on_grid = len(instr.desc.obj_set) > 0
                is_valid = obj_to_drop_on_grid or carried_expected_obj
            else:
                is_valid = len(instr.desc.obj_set) > 0

            if is_valid:
                if isinstance(instr,  PickupInstr):
                    instr.preCarrying = env.carrying
                if isinstance(instr, DropNextInstr):
                    instr.preCarrying = env.carrying
                    instr.initially_carried_world_obj = env.carrying
                if isinstance(instr, DropNextNothingInstr):
                    instr.preCarrying = env.carrying
                    if (env.carrying is not None) and instr.desc.type == env.carrying.type and instr.desc.color == env.carrying.color:
                        instr.initially_carried_world_obj = env.carrying
                valid_subgoals.append(subgoal)

              
        return valid_subgoals

    def update_current_valid_subgoals(self, env):
        # workaround: replace self.initial_valid_subgoals with self.all_subgoals to update the current
        # valid subgoals to work around the bug that subgoals related to carried and hidden objects are not
        # considered during the mission because they are excluded in the list of initial current sugboals
        self.current_valid_subgoals = self.filter_valid_subgoals(self.all_subgoals, env)

    def check_completed_subgoals(self, action, env):
        """
        Check if any valid low-level instructions are completed after performing the 'action'
        """
        completed_subgoals = []
        for subgoal_idx, subgoal_instr in self.current_valid_subgoals:
            result = subgoal_instr.verify(action)
            if result == 'success':
                completed_subgoals.append(subgoal_idx)

        # When the grid is changed, the valid subgoal instructions need to be updated
        if env.grid_changed:
            self.update_current_valid_subgoals(env)

        return completed_subgoals     

    def get_completed_subgoals_msg(self, completed_subgoals, debug=False):
        msg = []
        for idx in completed_subgoals:
            if debug:
                msg.append(f"[SG{idx}] "+self.all_subgoals[idx][1].instr_desc)
            else:
                msg.append(self.all_subgoals[idx][1].instr_desc)
        msg = ", ".join(msg) + "!"
        return msg 

class ObjDesc:
    """
    Description of a set of objects in an environment
    """

    def __init__(self, type, color=None, loc=None):
        assert type in [None, *OBJ_TYPES], type
        assert color in [None, *COLOR_NAMES], color
        assert loc in [None, *LOC_NAMES], loc

        self.color = color
        self.type = type
        self.loc = loc

        # Set of objects possibly matching the description
        # Include objects hidden in boxes
        self.obj_set = []

        # Set of initial object positions
        # It will be updated to track only Onboard and visible objects
        self.obj_poss = []

    def __repr__(self):
        return "{} {} {}".format(self.color, self.type, self.loc)

    def surface(self, env, is_carried=False):
        """
        Generate a natural language representation of the object description
        """

        if self.type:
            s = str(self.type)
        else:
            s = 'object'

        if self.color:
            s = self.color + ' ' + s

        if self.loc:
            if self.loc == 'front':
                s = s + ' in front of you'
            elif self.loc == 'behind':
                s = s + ' behind you'
            else:
                s = s + ' on your ' + self.loc

        self.find_matching_objs(env)
        if is_carried:
            s = 'the ' + s
        else:
            assert len(self.obj_set) > 0, "no object matching description"

            # Singular vs plural
            if len(self.obj_set) > 1:
                s = 'a ' + s
            else:
                s = 'the ' + s

        return s

    def find_matching_objs(self, env, use_location=True):
        """
        Find the set of objects matching the description and their positions.
        When use_location is False, we only update the positions of already tracked objects, without taking into account
        the location of the object. e.g. A ball that was on "your right" initially will still be tracked as being "on
        your right" when you move.

        Check the object hidden inside a box and added matched hidden objects to the obj_set
        Assume there is as most one hidden level. That is, the scenario that an object hides inside a box which hides another box is not considered.
        """

        if use_location:
            self.obj_set = []
            # otherwise we keep the same obj_set

        self.obj_poss = []

        agent_room = env.room_from_pos(*env.agent_pos)

        for i in range(env.grid.width):
            for j in range(env.grid.height):
                cell = env.grid.get(i, j)
                if cell is None:
                    continue

                if not use_location:
                    # we should keep tracking the same objects initially tracked only
                    already_tracked = any([cell is obj for obj in self.obj_set])
                    if not already_tracked:
                        continue

                matched_obj = None
                macthed_obj_hidden = False
                # If the current cell is box, check the object hidden inside a box.
                # Notes: it assumes that the box and the hidden object are the same and thus will not both match the target object.
                if cell.type == 'box' and cell.contains is not None:
                    if self.type is not None and cell.contains.type == self.type:
                        if self.color is not None and cell.contains.color == self.color:
                            matched_obj = cell.contains
                            macthed_obj_hidden = True

                if matched_obj is None:
                    # Check if object's type matches description
                    if self.type is not None and cell.type == self.type:
                        # Check if object's color matches description
                        if self.color is not None and cell.color == self.color:
                            matched_obj = cell

                if matched_obj is None:
                    continue

                # Check if object's position matches description
                if use_location and self.loc in ["left", "right", "front", "behind"]:
                    # Locations apply only to objects in the same room
                    # the agent starts in
                    if not agent_room.pos_inside(i, j):
                        continue

                    # Direction from the agent to the object
                    v = (i - env.agent_pos[0], j - env.agent_pos[1])

                    # (d1, d2) is an oriented orthonormal basis
                    d1 = DIR_TO_VEC[env.agent_dir]
                    d2 = (-d1[1], d1[0])

                    # Check if object's position matches with location
                    pos_matches = {
                        "left": dot_product(v, d2) < 0,
                        "right": dot_product(v, d2) > 0,
                        "front": dot_product(v, d1) > 0,
                        "behind": dot_product(v, d1) < 0
                    }

                    if not (pos_matches[self.loc]):
                        continue

                if use_location:
                    self.obj_set.append(matched_obj)

                # self.obj_poss stores the positions of tracked objects that on board and visible.
                if not macthed_obj_hidden:
                    self.obj_poss.append((i, j))

        return self.obj_set, self.obj_poss


class Instr:
    """
    Base class for all instructions in the baby language
    """

    def __init__(self):
        self.env = None
        self.instr_desc = ""

    def surface(self, env):
        """
        Produce a natural language representation of the instruction
        """

        raise NotImplementedError

    def reset_verifier(self, env):
        """
        Must be called at the beginning of the episode
        """

        self.env = env

    def verify(self, action):
        """
        Verify if the task described by the instruction is incomplete,
        complete with success or failed. The return value is a string,
        one of: 'success', 'failure' or 'continue'.
        """

        raise NotImplementedError

    def update_objs_poss(self):
        """
        Update the position of objects present in the instruction if needed
        """
        potential_objects = ('desc', 'desc_move', 'desc_fixed')
        for attr in potential_objects:
            if hasattr(self, attr):
                getattr(self, attr).find_matching_objs(self.env, use_location=False)


class ActionInstr(Instr):
    """
    Base class for all action instructions (clauses)
    """

    def __init__(self):
        super().__init__()

        # Indicates that the action was completed on the last step
        self.lastStepMatch = False

    def verify(self, action):
        """
        Verifies actions, with and without the done action.
        """

        if not use_done_actions:
            return self.verify_action(action)

        if action == self.env.actions.done:
            if self.lastStepMatch:
                return 'success'
            return 'failure'

        res = self.verify_action(action)
        self.lastStepMatch = (res == 'success')

    def verify_action(self):
        """
        Each action instruction class should implement this method
        to verify the action.
        """

        raise NotImplementedError


class OpenInstr(ActionInstr):
    def __init__(self, obj_desc, strict=False):
        super().__init__()
        assert obj_desc.type == 'door'
        self.desc = obj_desc
        self.strict = strict

    def surface(self, env):
        return 'open ' + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify_action(self, action):
        # Only verify when the toggle action is performed
        if action != self.env.actions.toggle:
            return 'continue'

        # Get the contents of the cell in front of the agent
        front_cell = self.env.grid.get(*self.env.front_pos)

        for door in self.desc.obj_set:
            if front_cell and front_cell is door and door.is_open:
                return 'success'

        # If in strict mode and the wrong door is opened, failure
        if self.strict:
            if front_cell and front_cell.type == 'door':
                return 'failure'

        return 'continue'


class GoToInstr(ActionInstr):
    """
    Go next to (and look towards) an object matching a given description
    eg: go to the door
    """

    def __init__(self, obj_desc):
        super().__init__()
        self.desc = obj_desc

    def surface(self, env):
        return 'go to ' + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify_action(self, action):

        # Only verify when the performed action is: left, right or forward 
        if action != self.env.actions.forward and action != self.env.actions.left and action != self.env.actions.right:
            return 'continue'

        # For each object position
        for pos in self.desc.obj_poss:
            # If the agent is next to (and facing) the object
            if np.array_equal(pos, self.env.front_pos):
                return 'success'

        return 'continue'

class OpenBoxInstr(ActionInstr):
    def __init__(self, obj_desc, strict=False):
        super().__init__()
        assert obj_desc.type == 'box'
        self.desc = obj_desc
        self.strict = strict

    def surface(self, env):
        return 'open ' + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify_action(self, action):
        # Only verify when the toggle action is performed
        if action != self.env.actions.toggle:
            return 'continue'

        for box in self.desc.obj_set:
            if self.env.box_opened and self.env.box_opened is box:
                return 'success'

        # If in strict mode and the wrong door is opened, failure
        if self.strict:
            if self.env.box_opened and self.env.box_opened.type == 'box':
                return 'failure'

        return 'continue'

class PickupInstr(ActionInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def __init__(self, obj_desc, strict=False):
        super().__init__()
        assert obj_desc.type != 'door'
        self.desc = obj_desc
        self.strict = strict

    def surface(self, env):
        return 'pick up ' + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # Object previously being carried
        self.preCarrying = None

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify_action(self, action):
        # To keep track of what was carried at the last time step
        preCarrying = self.preCarrying
        self.preCarrying = self.env.carrying

        # Only verify when the pickup action is performed
        if action != self.env.actions.pickup:
            return 'continue'

        for obj in self.desc.obj_set:
            if preCarrying is None and self.env.carrying is obj:
                return 'success'

        # If in strict mode and the wrong door object is picked up, failure
        if self.strict:
            if self.env.carrying:
                return 'failure'

        self.preCarrying = self.env.carrying

        return 'continue'


class PassInstr(ActionInstr):
    """
    Go through an opened door
    eg: pass the blue door
    """
    def __init__(self, obj_desc, strict=False):
        super().__init__()
        assert obj_desc.type == 'door'
        self.desc = obj_desc
        self.strict = strict

        self.doorApproached = None
        self.insideDoor = False
        # It is used to track the agent' direction relative
        # to the door when it is inside the door. 
        # Starts to count when seld.insideDoor becomes True
        # Default value is 0.
        # self.rotateTimes%4 = 
        #   0  : the agent faces the new room
        #   2  : the agent faces the coming room
        #   1/3: the agent faces a wall that is in a direction
        #        perpendicular to the passway of the door.
        #        So, 'farward' action takes no effect.
        self.rotateTimes = 0       

    def surface(self, env):
        return 'pass ' + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # entered the door
        self.insideDoor = False
        self.rotateTimes = 0
        # A door is approached (faced by or entered) by the agent
        self.doorApproached = None

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify_action(self, action):

        front_cell = self.env.grid.get(*self.env.front_pos)

        if self.insideDoor:
            if action == self.env.actions.forward:
                if self.rotateTimes%4 == 0:
                    if front_cell is None: # the front_cell is 'empty'
                        passCorrectDoor = False
                        for door in self.desc.obj_set:
                            if self.doorApproached is door:
                                passCorrectDoor = True
                                break
                        self.doorApproached = None
                        self.insideDoor = False
                        self.rotateTimes = 0
                             
                        if passCorrectDoor:        
                            return 'success'

                        # If in strict mode, passing a wrong door leads to a failure
                        if self.strict:
                            return 'failure'

                    # else branch: the passway is blocked by some object

                elif self.rotateTimes%4 == 2:
                    self.doorApproached = None
                    self.insideDoor = False
                    self.rotateTimes = 0
                #else branch: inside the door and face a wall

            elif action == self.env.actions.left:
                self.rotateTimes -= 1
            elif action == self.env.actions.right:
                self.rotateTimes += 1

        else:
            if not self.doorApproached:
                if front_cell and front_cell.type == 'door' and front_cell.color == self.desc.color and front_cell.is_open:
                    self.doorApproached = front_cell
            else: # the approached door is open
                if action == self.env.actions.forward:
                    self.insideDoor = True
                elif action == self.env.actions.left or action == self.env.actions.right or action == self.env.actions.toggle:
                    self.doorApproached = None
        
        return 'continue'

class DropNextInstr(ActionInstr):
    """
    Drop the carried object next to another object. Assume the agent is carring an object.
    eg: put the red ball next to the blue key
    """

    """
    Parameters:
        obj_carried: ObjDesc instance. used to differentiate the usages between low-level instr and mission goal
        initially_carried_world_obj: WorldObj instance. used in gen_mission() to indicate the carried obj when the mission starts
    """
    def __init__(self, obj_carried, obj_fixed, initially_carried_world_obj=None, strict=False):
        super().__init__()
        assert not obj_carried or obj_carried.type != 'door'
        self.initially_carried_obj = obj_carried
        self.initially_carried_world_obj = initially_carried_world_obj
        self.desc = obj_fixed # the target object that the agent needs to put its carried one next to 
        self.strict = strict

    def surface(self, env):
        carried_obj_desc = ""
        if self.initially_carried_obj:
            carried_obj_desc = " " + self.initially_carried_obj.surface(env, is_carried=True)
        return 'drop' + carried_obj_desc+ ' next to ' + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        if self.initially_carried_world_obj is not None:
            self.preCarrying = self.initially_carried_world_obj
        else:
            self.preCarrying = None

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify_action(self, action):
        # To keep track of what was carried at the last time step
        preCarrying = self.preCarrying
        self.preCarrying = self.env.carrying

        # Only verify when the drop action is performed
        #   preCarrying and not self.preCarring: the previously carried object has been successfully dropped
        if action == self.env.actions.drop and preCarrying and (not self.preCarrying):
            if preCarrying==self.initially_carried_world_obj:
                pos_a = preCarrying.cur_pos

                for pos_b in self.desc.obj_poss:
                    if pos_next_to(pos_a, pos_b):
                        return 'success'

                # in strict mode, droping the carried object next to a wrong object
                if self.strict:
                    return 'failure'

        return 'continue'

class DropNotNextInstr(ActionInstr):
    """
    Drop the carried object not next to another object. Assume the agent is carring an object.
    eg: put the red ball not next to the blue key
    """

    """
    Parameters:
        obj_carried: ObjDesc instance. used to differentiate the usages between low-level instr and mission goal
        initially_carried_world_obj: WorldObj instance. used in gen_mission() to indicate the carried obj when the mission starts
    """
    def __init__(self, obj_carried, obj_fixed, initially_carried_world_obj=None, strict=False):
        super().__init__()
        assert not obj_carried or obj_carried.type != 'door'
        self.initially_carried_obj = obj_carried
        self.initially_carried_world_obj = initially_carried_world_obj
        self.desc = obj_fixed # the target object that the agent needs to put its carried one next to 
        self.strict = strict

    def surface(self, env):
        carried_obj_desc = ""
        if self.initially_carried_obj:
            carried_obj_desc = " " + self.initially_carried_obj.surface(env, is_carried=True)
        return 'drop' + carried_obj_desc+ ' not next to ' + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        if self.initially_carried_world_obj is not None:
            self.preCarrying = self.initially_carried_world_obj
        else:
            self.preCarrying = None

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify_action(self, action):
        # To keep track of what was carried at the last time step
        preCarrying = self.preCarrying
        self.preCarrying = self.env.carrying

        # Only verify when the drop action is performed
        #   preCarrying and not self.preCarring: the previously carried object has been successfully dropped
        if action == self.env.actions.drop and preCarrying and (not self.preCarrying):
            if preCarrying==self.initially_carried_world_obj:
                pos_a = preCarrying.cur_pos

                not_next_to_target_objs = True
                for pos_b in self.desc.obj_poss:
                    if pos_next_to(pos_a, pos_b):
                        not_next_to_target_objs = False
                        break

                if not_next_to_target_objs:
                    return 'success'

                # in strict mode, droping the carried object next to a wrong object
                if self.strict:
                    return 'failure'

        return 'continue'

class DropNextNothingInstr(ActionInstr):
    """
    Drop the carried object to a location where there are no other objects nearby.
    eg: put the red ball next to the blue key
    """

    # Two use cases: two types of initialization parameter values
    # Case 1 (used when initializing a set of low-level instructions)
    #   obj_carried is None, obj_to_drop is not None
    # Case 2 (used when initializing a mission in DropNextNothingLocal environment)
    #   obj_carried == obj_to_drop, they are not None
    def __init__(self, initially_carried_world_obj, obj_to_drop, strict=False):
        super().__init__()
        assert not initially_carried_world_obj or initially_carried_world_obj.type != 'door'
        assert obj_to_drop and obj_to_drop.type != 'door'

        self.desc = obj_to_drop
        self.initially_carried_world_obj=initially_carried_world_obj

        self.strict = strict

    # Note:
    # According to BabyAI language in the BabyAI paper, the grammer of 'put' action is
    #   put <DescNotDoor> next to <Desc>
    # So, the instruction patter, 'put <DescNotDoor>', may not be recognized by the BOT
    # agent provided in the BabyAI paper such that we can not use the BOT agent to solve
    # the level 'DropNextNothingLocal' for collecting demonstrations.
    # But it will not impact the process of training a RL agent for solving tasks in
    # DropNextNothingLocal environments.
    # The issue can be solved by supplementing the 'put' grammar in BabyAI language
    # and updating the BOT.
    def surface(self, env):
        return 'drop ' + self.desc.surface(env, is_carried=True)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        if self.initially_carried_world_obj is not None:
            self.preCarrying = self.initially_carried_world_obj
        else:
            self.preCarrying = None

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify_action(self, action):
        # To keep track of what was carried at the last time step
        preCarrying = self.preCarrying
        self.preCarrying = self.env.carrying

        # Only verify when the drop action is performed
        #   preCarrying and not self.preCarring: the previously carried object has been successfully dropped
        if action == self.env.actions.drop and preCarrying and (not self.preCarrying):
            # the droped object matches that in the instruction
            if preCarrying == self.initially_carried_world_obj:
            #if preCarrying.type == self.desc.type and preCarrying.color == self.desc.color:
                # Note:
                # The agent is next to the dropped object on the grid, but it is not captured by the grid but by the env.
                # So, the cell where the agent locates is empty from the view of the grid. That is, the corresponding
                # nearby cell is None.
                nearby_cells = self.env.grid.get_nearby_cells(preCarrying.cur_pos)

                # If the dropped object is not nearby another other objects (excluding walls and the agent),
                # this instruction is verifed as success
                for cell in nearby_cells:
                    # If there is another object next to the dropped object,
                    # the instruction is not successful.
                    if cell and cell.type != 'wall':
                        if self.strict:
                            return 'failure'
                        else:
                            return 'continue'
            
                return 'success'

        return 'continue'

class PutNextInstr(ActionInstr):
    """
    Put an object next to another object
    eg: put the red ball next to the blue key
    """

    def __init__(self, obj_move, obj_fixed, strict=False):
        super().__init__()
        assert obj_move.type != 'door'
        self.desc_move = obj_move
        self.desc_fixed = obj_fixed
        self.strict = strict

    def surface(self, env):
        return 'put ' + self.desc_move.surface(env) + ' next to ' + self.desc_fixed.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # Object previously being carried
        self.preCarrying = None

        # Identify set of possible matching objects in the environment
        self.desc_move.find_matching_objs(env)
        self.desc_fixed.find_matching_objs(env)

    def objs_next(self):
        """
        Check if the objects are next to each other
        This is used for rejection sampling
        """

        for obj_a in self.desc_move.obj_set:
            # an obj in the obj_set has cur_poss of None indicates it hides in a box
            if obj_a.cur_pos is not None:
                pos_a = obj_a.cur_pos

                for pos_b in self.desc_fixed.obj_poss:
                    if pos_next_to(pos_a, pos_b):
                        return True
        return False

    def verify_action(self, action):
        # To keep track of what was carried at the last time step
        preCarrying = self.preCarrying
        self.preCarrying = self.env.carrying

        # In strict mode, picking up the wrong object fails
        if self.strict:
            if action == self.env.actions.pickup and self.env.carrying:
                return 'failure'

        # Only verify when the drop action is performed
        if action != self.env.actions.drop:
            return 'continue'

        for obj_a in self.desc_move.obj_set:
            if preCarrying is not obj_a:
                continue

            pos_a = obj_a.cur_pos

            for pos_b in self.desc_fixed.obj_poss:
                if pos_next_to(pos_a, pos_b):
                    return 'success'

        return 'continue'


class SeqInstr(Instr):
    """
    Base class for sequencing instructions (before, after, and)
    """

    def __init__(self, instr_a, instr_b, strict=False):
        assert isinstance(instr_a, ActionInstr) or isinstance(instr_a, AndInstr)
        assert isinstance(instr_b, ActionInstr) or isinstance(instr_b, AndInstr)
        self.instr_a = instr_a
        self.instr_b = instr_b
        self.strict = strict


class BeforeInstr(SeqInstr):
    """
    Sequence two instructions in order:
    eg: go to the red door then pick up the blue ball
    """

    def surface(self, env):
        return self.instr_a.surface(env) + ', then ' + self.instr_b.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)
        self.instr_a.reset_verifier(env)
        self.instr_b.reset_verifier(env)
        self.a_done = False
        self.b_done = False

    def verify(self, action):
        if self.a_done == 'success':
            self.b_done = self.instr_b.verify(action)

            if self.b_done == 'failure':
                return 'failure'

            if self.b_done == 'success':
                return 'success'
        else:
            self.a_done = self.instr_a.verify(action)
            if self.a_done == 'failure':
                return 'failure'

            if self.a_done == 'success':
                return self.verify(action)

            # In strict mode, completing b first means failure
            if self.strict:
                if self.instr_b.verify(action) == 'success':
                    return 'failure'

        return 'continue'


class AfterInstr(SeqInstr):
    """
    Sequence two instructions in reverse order:
    eg: go to the red door after you pick up the blue ball
    """

    def surface(self, env):
        return self.instr_a.surface(env) + ' after you ' + self.instr_b.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)
        self.instr_a.reset_verifier(env)
        self.instr_b.reset_verifier(env)
        self.a_done = False
        self.b_done = False

    def verify(self, action):
        if self.b_done == 'success':
            self.a_done = self.instr_a.verify(action)

            if self.a_done == 'success':
                return 'success'

            if self.a_done == 'failure':
                return 'failure'
        else:
            self.b_done = self.instr_b.verify(action)
            if self.b_done == 'failure':
                return 'failure'

            if self.b_done == 'success':
                return self.verify(action)

            # In strict mode, completing a first means failure
            if self.strict:
                if self.instr_a.verify(action) == 'success':
                    return 'failure'

        return 'continue'


class AndInstr(SeqInstr):
    """
    Conjunction of two actions, both can be completed in any other
    eg: go to the red door and pick up the blue ball
    """

    def __init__(self, instr_a, instr_b, strict=False):
        assert isinstance(instr_a, ActionInstr)
        assert isinstance(instr_b, ActionInstr)
        super().__init__(instr_a, instr_b, strict)

    def surface(self, env):
        return self.instr_a.surface(env) + ' and ' + self.instr_b.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)
        self.instr_a.reset_verifier(env)
        self.instr_b.reset_verifier(env)
        self.a_done = False
        self.b_done = False

    def verify(self, action):
        if self.a_done != 'success':
            self.a_done = self.instr_a.verify(action)

        if self.b_done != 'success':
            self.b_done = self.instr_b.verify(action)

        if use_done_actions and action is self.env.actions.done:
            if self.a_done == 'failure' and self.b_done == 'failure':
                return 'failure'

        if self.a_done == 'success' and self.b_done == 'success':
            return 'success'

        return 'continue'
