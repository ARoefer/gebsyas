bl_info = {
    "name": "GMM scene utils",
    "category": "Object",
}

import bpy
from bpy.props import (
        BoolProperty,
        FloatProperty,
        StringProperty,
        EnumProperty,
        )
from bpy_extras.io_utils import (
        ExportHelper,
        path_reference_mode)


def is_atom_or_alist(obj):
    if not is_atom(obj) and type(obj) == list or type(obj) == set or type(obj) == tuple:
        return max([is_atom(x) for x in obj]) if len(obj) > 0 else True
    return is_atom(obj)

def is_atom(obj):
    return type(obj) != list and type(obj) != set and type(obj) != tuple and type(obj) != dict

def gen_yaml_str(obj):
    if type(obj) == list or type(obj) == set or type(obj) == tuple:
        if len(obj) == 0:
            return '[]'
        if max([is_atom(x) for x in obj]):
            return '[{}]'.format(', '.join([gen_yaml_str(x) for x in obj]))
        substrs = []
        for x in obj:
            substrs.append('- {}'.format(gen_yaml_str(x)).replace('\n', '\n  '))
        return '\n'.join(substrs)
    elif type(obj) == dict:
        if len(obj) == 0:
            return '{}'
        return '\n'.join(['{}: {}'.format(k, str(v)) if is_atom_or_alist(v) else '{}:\n  {}'.format(k, gen_yaml_str(v).replace('\n', '\n  ')) for k, v in obj.items()])
    else:
        return str(obj)

def hsva_to_rgba(h, s, v, a):
    h_i = int(round(h*6))
    f = h*6 - h_i
    p = v * (1 - s)
    q = v * (1 - f*s)
    t = v * (1 - (1 - f) * s)
    if h_i==0:
        return [v, t, p, a]
    elif h_i==1:
        return [q, v, p, a]
    elif h_i==2:
        return [p, v, t, a]
    elif h_i==3:
        return [p, q, v, a]
    elif h_i==4:
        return [t, p, v, a]
    elif h_i==5:
        return [v, p, q, a]
    return [1,1,1,a]

def indent(str, level):
    return str.replace('\t', '  '*level)

def pose_data(obj):
    obj.rotation_mode = 'QUATERNION'
    pos  = list(obj.location)
    quat = [obj.rotation_quaternion.x, obj.rotation_quaternion.y, obj.rotation_quaternion.z, obj.rotation_quaternion.w]
    return {'position': pos,
            'rotation': quat}

def obj_data(obj):                
    type = obj.data.name.split('.')[0].lower()
    color = (0.4, 0.4, 0.4, 1.0) if obj.location[2] < 0 else hsva_to_rgba(min(obj.location[2] / 2.0, 1.0) * 0.65, 1, 1, 1)
    color = '[{}]'.format(', '.join([str(x) for x in color]))
    extents = list(obj.dimensions)
    mass = obj['mass'] if 'mass' in obj else 0.0
    
    return {'name': obj.name, 
            'type': 'rigid_body', 
            'geom_type': type, 
            'initial_pose': pose_data(obj), 
            'color': color, 
            'mass': mass,
            'extents': extents,
            'radius': extents[0]*0.5,
            'height': extents[2]}

def get_gc_weight(obj):
    for c in obj.children:
        if 'GMM_Weight' in c:
            return c.scale.z
    return 0.0

def set_gc_weight(obj, value):
    for c in obj.children:
        if 'GMM_Weight' in c:
            c.scale = (value,)*3

def get_gc_spread(obj):
    for c in obj.children:
        if 'GMM_Spread' in c:
            return c.scale
    return (0.0, 0.0, 0.0)

def cov_pose_data(obj):
    spread = get_gc_spread(obj)
    
    cov_str = '[{}, 0, 0, 0, 0, 0, 0, {}, 0, 0, 0, 0, 0, 0, {}, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'.format(spread[0], spread[1], spread[2])
    return {'cov': cov_str,
            'weight': get_gc_weight(obj),
            'pose': pose_data(obj)}

class ExportScene(bpy.types.Operator, ExportHelper):
    """My Object Moving Script"""      # blender will use this as a tooltip for menu items and buttons.
    bl_idname = "export_scene.gmm"        # unique identifier for buttons and menu items to reference.
    bl_label = "Export IAI-Simulation Scene"         # display name in the interface.
    bl_options = {'PRESET'}  # enable undo for the operator.

    filename_ext = ".yaml"
    filter_glob = StringProperty(
            default="*.yaml",
            options={'HIDDEN'},
            )

    use_gmm = BoolProperty(
            name="Write GMMs",
            default=True)

    path_mode = path_reference_mode

    check_extension = True

    def execute(self, context):        # execute() is called by blender when running the operator.
        C = context
        objs = []

        gmm_objs = {}
        file_dict = {'world': {'constraints': [], 
                               'objects': []},
                     'gmm_objects': [],
                     'gravity': [0, 0, -9.81],
                     'tick_rate': 50}

        for name, v in C.scene.objects.items():
            if not v.is_visible(C.scene) or v.data is None:
                continue
            
            if '.' in name:
                parts = name.split('.')
                try:
                    parts[-1] = str(int(parts[-1]))
                except:
                    pass
                v.name = ''.join(parts)
            
            if 'GMM_Root' in v: 
                if v not in gmm_objs:
                    gmm_objs[v] = []
            elif 'GMM_Object' in v:
                if v['GMM_Object'] not in gmm_objs:
                    gmm_objs[v['GMM_Object']] = []
                gmm_objs[v['GMM_Object']].append(v)
                print('Identified {} as GC of {}'.format(v.name, v['GMM_Object'].name))
            elif v.data.name.split('.')[0] in ['Box', 'Cylinder', 'Sphere']:
                file_dict['world']['objects'].append(obj_data(v))  
        
        print(gmm_objs)
        
        for gmm_obj, gmcs in gmm_objs.items():
            # Normalize component weights
            sum_weight = get_gc_weight(gmm_obj)
            for gc in gmcs:
                sum_weight += get_gc_weight(gc)
            print('Total weight for {}: {}'.format(gmm_obj.name, sum_weight))
            weight_scale = 1.0 / sum_weight
            set_gc_weight(gmm_obj, get_gc_weight(gmm_obj) * weight_scale)
            for gc in gmcs:
                set_gc_weight(gc, get_gc_weight(gc) * weight_scale)
        
            odata = obj_data(gmm_obj)
            odata['gmm'] = [cov_pose_data(gmm_obj)] + [cov_pose_data(gc) for gc in gmcs]
            file_dict['gmm_objects'].append(odata['name'])
            file_dict['world']['objects'].append(odata)

        f = open(self.filepath, 'w')
        f.write(gen_yaml_str(file_dict))
        f.close()
        return {'FINISHED'}            # this lets blender know the operator finished successfully.

class CreateGMMComponent(bpy.types.Operator):
    """My Object Moving Script"""      # blender will use this as a tooltip for menu items and buttons.
    bl_idname = "gmm_tools.gmm"        # unique identifier for buttons and menu items to reference.
    bl_label = "Create Gaussian Component for Object"         # display name in the interface.
    bl_options = {'UNDO', 'REGISTER'}  # enable undo for the operator.

    def execute(self, context): 
        if len(context.selected_objects) == 0 or context.active_object.type != 'MESH':
            return {'FINISHED'}
        
        ao = context.active_object
        if len(context.selected_objects) > 1:
            for o in context.selected_objects:
                o.select = False
        
        if 'GMM_Root' not in ao and 'GMM_Object' not in ao:
            ao['GMM_Root'] = True
            
            bpy.ops.object.add(radius=0.5)
            gmm_spread = context.active_object
            gmm_spread.location = ao.location
            gmm_spread['GMM_Spread'] = True
            gmm_spread.empty_draw_type = 'SPHERE'
            bpy.ops.object.add(radius=1)
            gmm_weight = context.active_object
            gmm_weight.location = ao.location
            gmm_weight['GMM_Weight'] = True
            gmm_weight.empty_draw_type = 'SINGLE_ARROW'
            gmm_spread.select = True
            gmm_weight.select = True
            context.scene.objects.active = ao
            bpy.ops.object.parent_set()
        else:
            for c in ao.children:
                if 'GMM_Spread' in c or 'GMM_Weight' in c:
                    c.select = True
        ao.select = True
        context.scene.objects.active = ao
        bpy.ops.object.duplicate(linked=True)
        gc = context.active_object
        del gc['GMM_Root']
        gc['GMM_Object'] = ao
        gc.draw_type = 'WIRE'
        bpy.ops.transform.translate()

        return {'FINISHED'}

def menu_func_export(self, context):
    self.layout.operator(ExportScene.bl_idname, text="GMM-Map (.yaml)")

def register():
    bpy.utils.register_class(ExportScene)
    bpy.utils.register_class(CreateGMMComponent)
    bpy.types.INFO_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(ExportScene)
    bpy.utils.unregister_class(CreateGMMComponent)
    bpy.types.INFO_MT_file_export.remove(menu_func_export)


# This allows you to run the script directly from blenders text editor
# to test the addon without having to install it.
if __name__ == "__main__":
    register()
    
    #unregister()
    