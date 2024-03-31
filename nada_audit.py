"""
CLI-/browser-compatible tool that statically audits Nada programs via syntax
checking, type inference, and complexity/cost analysis.
"""
from __future__ import annotations
from typing import Union, List, Tuple
import argparse
import ast
import asttokens
import richreports
import parsial

class Rule(Exception):
    pass

class SyntaxRestriction(Rule):
    pass

class TypeErrorRoot(TypeError):
    """
    This class represents type errors that are not caused by other type errors.
    """

def typeerror_demote(t):
    if isinstance(t, TypeErrorRoot):
        return TypeError(str(t))
    return t

class Abstract:
    inputs = None
    outputs = None

    @staticmethod
    def initialize(context=None):
        Abstract.inputs = []
        Abstract.outputs = []
        Abstract.context = context if context is not None else {}

    def __init__(self, cls=None):
        self.value = None
        if cls is not None:
            self.__class__ = cls

class Party(Abstract):
    def __init__(self: Party, name: str):
        if not isinstance(name, str):
            raise TypeError('name parameter must be a string')

        self.name = name

class Input(Abstract):
    def __init__(self: Input, name: str, party: Party):
        type(self).inputs.append(self)
    
        if not isinstance(name, str):
            raise TypeError('name parameter must be a string')

        if not isinstance(name, str):
            raise TypeError('party parameter must be a Party object')

        self.name = name
        self.party = party

    def value(self):
        return self.context.get(self.name, None)

class Output(Abstract):
    def __init__(
            self: Output,
            value: Union[PublicInteger, SecretInteger],
            name: str,
            party: Party
        ):
        type(self).outputs.append(self)

        if not isinstance(value, (PublicInteger, SecretInteger)):
            raise TypeError('output value must be a PublicInteger or a SecretInteger')

        if not isinstance(name, str):
            raise TypeError('name parameter must be a string')

        if not isinstance(name, str):
            raise TypeError('party parameter must be a Party object')

        self.value = value
        self.name = name
        self.party = party

class PublicInteger(Abstract):
    def __init__(self: Output, input: Input = None, value: int = None):
        self.input = input
        self.value = self.input.value() if input is not None else value

    def __add__(self: PublicInteger, other: Union[PublicInteger, SecretInteger]):
        """
        The table below presents the output type for each combination
        of argument types.

        +-------------------+-------------------+-------------------+
        |     ``self``      |     ``other``     |    **result**     |
        +-------------------+-------------------+-------------------+
        | ``PublicInteger`` | ``PublicInteger`` | ``PublicInteger`` |
        +-------------------+-------------------+-------------------+
        | ``PublicInteger`` | ``SecretInteger`` | ``SecretInteger`` |
        +-------------------+-------------------+-------------------+
        | ``SecretInteger`` | ``PublicInteger`` | ``SecretInteger`` |
        +-------------------+-------------------+-------------------+
        | ``SecretInteger`` | ``SecretInteger`` | ``SecretInteger`` |
        +-------------------+-------------------+-------------------+
        """
        if other == 0: # Base case for ``sum``.
            result = Abstract(PublicInteger)
            result.value = self.value
            return result
            
        if isinstance(other, PublicInteger):
            result = Abstract(PublicInteger)
            result.value = None
            if self.value is not None and other.value is not None:
                result.value = self.value + other.value
            return result

        if isinstance(other, SecretInteger):
            result = Abstract(SecretInteger)
            result.value = None
            if self.value is not None and other.value is not None:
                result.value = self.value + other.value
            return result

        raise TypeError('expecting PublicInteger or SecretInteger')

    def __radd__(self: PublicInteger, other: Union[PublicInteger, SecretInteger]):
        return self.__add__(other)

    def __mul__(self: PublicInteger, other: Union[PublicInteger, SecretInteger]):
        """
        The table below presents the output type for each combination
        of argument types.

        +-------------------+-------------------+-------------------+
        |     ``self``      |     ``other``     |    **result**     |
        +-------------------+-------------------+-------------------+
        | ``PublicInteger`` | ``PublicInteger`` | ``PublicInteger`` |
        +-------------------+-------------------+-------------------+
        | ``PublicInteger`` | ``SecretInteger`` | ``SecretInteger`` |
        +-------------------+-------------------+-------------------+
        | ``SecretInteger`` | ``PublicInteger`` | ``SecretInteger`` |
        +-------------------+-------------------+-------------------+
        | ``SecretInteger`` | ``SecretInteger`` | ``SecretInteger`` |
        +-------------------+-------------------+-------------------+
        """
        if isinstance(other, PublicInteger):
            result = Abstract(PublicInteger)
            result.value = None
            if self.value is not None and other.value is not None:
                result.value = self.value * other.value
            return result

        if isinstance(other, SecretInteger):
            result = Abstract(SecretInteger)
            result.value = None
            if self.value is not None and other.value is not None:
                result.value = self.value * other.value
            return result

        raise TypeError('expecting PublicInteger or SecretInteger')

    def __rmul__(self: PublicInteger, other: Union[PublicInteger, SecretInteger]):
        return self.__mul__(other)

class SecretInteger(Abstract):
    def __init__(self: Output, input: Input = None, value: int = None):
        self.input = input
        self.value = self.input.value() if input is not None else value

    def __add__(self: SecretInteger, other: Union[PublicInteger, SecretInteger]):
        """
        The table below presents the output type for each combination
        of argument types.

        +-------------------+-------------------+-------------------+
        |     ``self``      |     ``other``     |    **result**     |
        +-------------------+-------------------+-------------------+
        | ``PublicInteger`` | ``PublicInteger`` | ``PublicInteger`` |
        +-------------------+-------------------+-------------------+
        | ``PublicInteger`` | ``SecretInteger`` | ``SecretInteger`` |
        +-------------------+-------------------+-------------------+
        | ``SecretInteger`` | ``PublicInteger`` | ``SecretInteger`` |
        +-------------------+-------------------+-------------------+
        | ``SecretInteger`` | ``SecretInteger`` | ``SecretInteger`` |
        +-------------------+-------------------+-------------------+
        """
        if other == 0: # Base case for ``sum``.
            result = Abstract(SecretInteger)
            result.value = self.value
            return result

        if not isinstance(other, (PublicInteger, SecretInteger)):
            raise TypeError('expecting PublicInteger or SecretInteger')

        result = Abstract(SecretInteger)
        result.value = None
        if self.value is not None and other.value is not None:
            result.value = self.value + other.value
        return result

    def __radd__(self: PublicInteger, other: Union[PublicInteger, SecretInteger]):
        return self.__add__(other)

    def __mul__(self: SecretInteger, other: Union[PublicInteger, SecretInteger]):
        """
        The table below presents the output type for each combination
        of argument types.

        +-------------------+-------------------+-------------------+
        |     ``self``      |     ``other``     |    **result**     |
        +-------------------+-------------------+-------------------+
        | ``PublicInteger`` | ``PublicInteger`` | ``PublicInteger`` |
        +-------------------+-------------------+-------------------+
        | ``PublicInteger`` | ``SecretInteger`` | ``SecretInteger`` |
        +-------------------+-------------------+-------------------+
        | ``SecretInteger`` | ``PublicInteger`` | ``SecretInteger`` |
        +-------------------+-------------------+-------------------+
        | ``SecretInteger`` | ``SecretInteger`` | ``SecretInteger`` |
        +-------------------+-------------------+-------------------+
        """
        if not isinstance(other, (PublicInteger, SecretInteger)):
            raise TypeError('expecting PublicInteger or SecretInteger')

        result = Abstract(SecretInteger)
        result.value = None
        if self.value is not None and other.value is not None:
            result.value = self.value * other.value
        return result

    def __rmul__(self: PublicInteger, other: Union[PublicInteger, SecretInteger]):
        return self.__mul__(other)

def _parse(source: str) -> Tuple[asttokens.ASTTokens, List[int]]:
    lines = source.split('\n')
    (_, slices) = parsial.parsial(ast.parse)(source)
    lines_ = [l[s] for (l, s) in zip(lines, slices)]
    skips = [i for i in range(len(lines)) if len(lines[i]) != len(lines_[i])]
    return (asttokens.ASTTokens('\n'.join(lines_), parse=True), skips)

def _audits(node, key, value=None, default=None, delete=False):
    if not hasattr(node, '_audits'):
        setattr(node, '_audits', {})

    if value is None:
        value = node._audits.get(key, default)
        if delete and value in node._audits:
            del node._audits[key]
        return value

    node._audits[key] = value

def _types_binop_mult_add(t_l, t_r):
    t = TypeErrorRoot('arguments must be public or secret integers')
    if (t_l, t_r) == (SecretInteger, SecretInteger):
        t = SecretInteger
    elif (t_l, t_r) == (PublicInteger, SecretInteger):
        t = SecretInteger
    elif (t_l, t_r) == (SecretInteger, PublicInteger):
        t = SecretInteger
    elif (t_l, t_r) == (PublicInteger, PublicInteger):
        t = PublicInteger
    return t

def rules(a):
    prohibited = True

    if isinstance(a, ast.Module):
        prohibited = False
        for a_ in a.body:
            rules(a_)

    elif isinstance(a, ast.ImportFrom):
        if (
            a.module == 'nada_audit' and
            len(a.names) == 1 and a.names[0].name == '*' and a.names[0].asname is None and
            a.level == 0
        ):
            prohibited = False

    elif isinstance(a, ast.FunctionDef):
        if a.name == 'nada_main':
            prohibited = False
            for a_ in a.body:
                rules(a_)

    elif isinstance(a, (ast.Assign, ast.AnnAssign)):
        prohibited = False
        rules(a.value)

    elif isinstance(a, ast.Return):
        prohibited = False
        rules(a.value)

    elif isinstance(a, ast.Expr):
        prohibited = False
        rules(a.value)

    elif isinstance(a, ast.ListComp):
        prohibited = False
        rules(a.elt)
        for comprehension in a.generators:
            rules(comprehension.iter)

    elif isinstance(a, ast.Call):
        for a_ in a.args:
            rules(a_)
        for a_ in a.keywords:
            rules(a_.value)
        if isinstance(a.func, ast.Name):
            if a.func.id in (
                'range', 'str', 'sum',
                'Party', 'Input', 'Output',
                'PublicInteger', 'SecretInteger'
            ):
                prohibited = False
        elif isinstance(a.func, ast.Attribute):
            if a.func.attr == 'append':
                prohibited = False

    elif isinstance(a, ast.Subscript):
        prohibited = False
        rules(a.value)
        rules(a.slice)

    elif isinstance(a, ast.List):
        prohibited = False
        for a_ in a.elts:
            rules(a_)

    elif isinstance(a, ast.BinOp):
        if isinstance(a.op, ast.Mult):
            prohibited = False
            rules(a.left)
            rules(a.right)
        elif isinstance(a.op, ast.Add):
            prohibited = False
            rules(a.left)
            rules(a.right)

    elif isinstance(a, ast.Name):
        prohibited = False

    elif isinstance(a, ast.Constant):
        if isinstance(a.value, (str, int)):
            prohibited = False

    if prohibited: # None of the above cases matched.
        _audits(a, 'rules', SyntaxRestriction('use of this syntax is prohibited'))
        #if hasattr(a, 'iter_child_nodes'):
        #    for a_ in a.iter_child_nodes():
        #        if isinstance(_audits(a, 'rules'), SyntaxRestriction):
        #            _audits(a, 'rules', delete=True)

def types(a, env=None):
    env = {} if env is None else env

    # Handle cases in which the input node is a statement.
    if isinstance(a, ast.Module):
        for a_ in a.body:
            env = types(a_, env)
        return env

    elif isinstance(a, ast.FunctionDef):
        if a.name == 'nada_main':
            # Create a local copy of the environment. Only the
            # original environment passed to this invocation
            # is returned.
            env_ = dict(env)
            for a_ in a.body:
                env_ = types(a_, env_)
        return env

    elif isinstance(a, (ast.Assign, ast.AnnAssign)):
        types(a.value, env)
        t = _audits(a.value, 'types')

        if hasattr(a, 'annotation'):
            t = eval(ast.unparse(a.annotation))

        if t is not None:
            _audits(a, 'types', typeerror_demote(t))
            if not isinstance(t, TypeError):
                target = a.targets[0] if hasattr(a, 'targets') else a.target
                if isinstance(target, ast.Name):
                    var = target.id
                    env[var] = t

        return env

    elif isinstance(a, ast.Return):
        types(a.value, env)
        t = _audits(a.value, 'types')
        if t is not None:
            _audits(a, 'types', t)
        return env

    elif isinstance(a, ast.For):
        types(a.iter, env)
        var = a.target.id
        _audits(a.target, 'types', int)
        env[var] = int
        for a_ in a.body:
            env = types(a_, env)
        return env

    elif isinstance(a, ast.Expr):
        types(a.value, env)
        t = _audits(a.value, 'types')
        if t is not None:
            _audits(a, 'types', t)
        return env

    # Handle cases in which the input node is an expression.
    _audits(a, 'types', TypeError('type cannot be determined'))

    if isinstance(a, ast.ListComp):
        ts = {}
        for comprehension in a.generators:
            var = comprehension.target.id
            types(comprehension.iter, env)
            t_c = _audits(comprehension.iter, 'types')
            if t_c == range:
                ts[var] = int
                _audits(comprehension.target, 'types', int)
        env_ = dict(env)
        for (var, t_) in ts.items():
            env_[var] = t_
        types(a.elt, env_)
        t_e = _audits(a.elt, 'types')
        if t_e is not None and not isinstance(t_e, TypeError):
            _audits(a, 'types', list[t_e])

    elif isinstance(a, ast.Call):
        for a_ in a.args:
            types(a_, env)
        for a_ in a.keywords:
            types(a_.value, env)

        if isinstance(a.func, ast.Attribute):
            types(a.func.value, env)
            t = _audits(a.func.value, 'types')
            _audits(a.func, 'types', t)
            _audits(a, 'types', t)

        elif isinstance(a.func, ast.Name):
            if a.func.id == 'Party':
                t = TypeError('party requires name parameter')
                if (
                    (
                        len(a.args) == 0 and
                        len(a.keywords) == 1 and
                        a.keywords[0].arg == 'name' and
                        _audits(a.keywords[0].value, 'types') == str
                    )
                    or
                    (
                        len(a.args) == 1 and
                        len(a.keywords) == 0 and
                        _audits(a.args[0], 'types') == str
                    )
                ):
                    t = Party
                _audits(a, 'types', t)

            elif a.func.id == 'Input':
                t = TypeError('input requires name and party parameters')
                if (
                    (
                        len(a.args) == 2 and
                        _audits(a.args[0], 'types') == str and
                        _audits(a.args[1], 'types') == Party 
                    )
                    or
                    (
                        len(a.args) == 1 and
                        len(a.keywords) == 1 and
                        _audits(a.args[0], 'types') == str and
                        a.keywords[0].arg == 'party' and
                        _audits(a.keywords[0].value, 'types') == Party
                    )
                    or
                    (
                        len(a.args) == 0 and
                        len(a.keywords) == 2 and
                        a.keywords[0].arg == 'name' and
                        _audits(a.keywords[0].value, 'types') == str and
                        a.keywords[1].arg == 'party' and
                        _audits(a.keywords[1].value, 'types') == Party
                    )
                    or
                    (
                        len(a.args) == 0 and
                        len(a.keywords) == 2 and
                        a.keywords[1].arg == 'name' and
                        _audits(a.keywords[1].value, 'types') == str and
                        a.keywords[0].arg == 'party' and
                        _audits(a.keywords[0].value, 'types') == Party
                    )
                ):
                    t = Input
                _audits(a, 'types', t)

            elif a.func.id == 'Output':
                t = Output
                if (
                    len(a.args) != 3 or
                    len(a.keywords) != 0 or
                    _audits(a.args[0], 'types') not in (SecretInteger, PublicInteger) or
                    _audits(a.args[1], 'types') != str or
                    _audits(a.args[2], 'types') != Party
                ):
                    t = TypeError(
                        'output requires value, name, and party parameters'
                    )
                _audits(a, 'types', t)

            elif a.func.id == 'PublicInteger':
                t = PublicInteger
                if (
                    (len(a.args) != 1) or
                    _audits(a.args[0], 'types') != Input
                ):
                    t = TypeError('expecting single input argument')
                _audits(a, 'types', t)

            elif a.func.id == 'SecretInteger':
                t = SecretInteger
                if (
                    (len(a.args) != 1) or
                    _audits(a.args[0], 'types') != Input
                ):
                    t = TypeError('expecting single input argument')
                _audits(a, 'types', t)

            elif a.func.id == 'range':
                t = range
                if (
                    (len(a.args) != 1) or
                    _audits(a.args[0], 'types') != int
                ):
                    t = TypeError('expecting single integer argument')
                _audits(a, 'types', t)

            elif a.func.id == 'str':
                t = str
                if (
                    (len(a.args) != 1) or
                    _audits(a.args[0], 'types') != int
                ):
                    t = TypeError('expecting single integer argument')
                _audits(a, 'types', t)

            elif a.func.id == 'sum':
                t = SecretInteger
                if (
                    (len(a.args) != 1) or
                    _audits(a.args[0], 'types') != list[SecretInteger]
                ):
                    t = TypeError('expecting argument of type list[SecretInteger]')
                _audits(a, 'types', t)

    elif isinstance(a, ast.Subscript):
        types(a.value, env)
        types(a.slice, env)
        t_v = _audits(a.value, 'types')
        t_s = _audits(a.slice, 'types')
        t = TypeError('expecting list value and integer index')
        if t_v.__name__ == 'list' and t_s == int:
            if hasattr(t_v, '__args__') and len(t_v.__args__) == 1:
                t = t_v.__args__[0]
                _audits(a, 'types', t)

    elif isinstance(a, ast.List):
        for a_ in a.elts:
            types(a_, env)

        ts = [_audits(a_, 'types') for a_ in a.elts]
        t = TypeError('lists must contain elements that are all of the same type')
        if len(set(ts)) == 0:
           t = list
        elif len(set(ts)) == 1:
           t = list[ts[0]]
        _audits(a, 'types', t)

    elif isinstance(a, ast.BinOp):
        types(a.left, env)
        types(a.right, env)
        t_l = _audits(a.left, 'types')
        t_r = _audits(a.right, 'types')
        if isinstance(a.op, ast.Mult):
            t = _types_binop_mult_add(t_l, t_r)
            _audits(a, 'types', t)
        elif isinstance(a.op, ast.Add):
            t = TypeError('unsupported argument types')
            if t_l == str and t_r == str:
                t = str
            else:
                t = _types_binop_mult_add(t_l, t_r)
            _audits(a, 'types', t)

    elif isinstance(a, ast.Name):
        var = a.id
        _audits(
            a,
            'types',
            env[var] if var in env else TypeError('unbound variable: ' + var)
        )

    elif isinstance(a, ast.Constant):
        if isinstance(a.value, int):
            _audits(a, 'types', int)
        elif isinstance(a.value, str):
            _audits(a, 'types', str)

    return env # Always return the environment.

def _type_to_str(t):
    if hasattr(t, '__name__'):
        if t.__name__ == 'list':
            if hasattr(t, '__args__'):
                return 'list[' + _type_to_str(t.__args__[0]) + ']'
            return 'list'

        return str(t.__name__)
    
    return str('TypeError: ' + 'type cannot be determined')

def _enrich_from_type(report_, type_, start, end):
    if (
        type_ in (
            int, str, range,
            Party, Input, Output,
            PublicInteger, SecretInteger
        )
        or
        (
            hasattr(type_, '__name__') and
            type_.__name__ == 'list'
        )
    ):
        t_str = _type_to_str(type_)
        report_.enrich(
            start, end,
            '<span class="types-' + t_str + '">', '</span>',
            True
        )
    if isinstance(type_, (TypeError, TypeErrorRoot)):
        report_.enrich(
            start, end,
            '<span class="types-' + type_.__class__.__name__ + '">', '</span>',
            True
        )

def _enrich_syntaxrestriction(report_, r, start, end):
    report_.enrich(
        start, end,
        '<span class="rules-SyntaxRestriction">', '</span>',
        enrich_intermediate_lines=True,
        skip_whitespace=True
    )
    report_.enrich(
        start, end,
        '<span class="detail" data-detail="SyntaxRestriction: ' + str(r) + '">',
        '</span>',
        enrich_intermediate_lines=True,
        skip_whitespace=True
    )

def _enrich_keyword(report_, start, length):
    (start_line, start_column) = start
    report_.enrich(
        (start_line, start_column), (start_line, start_column + length),
        '<span class="keyword">', '</span>',
        True
    )

def _locations(report_, asttokens_, a):
    ((start_line, start_column), (end_line, end_column)) = \
        asttokens_.get_text_positions(a, True)
    
    # Skip any whitespace when determining the starting location.
    line = report_.lines[start_line - 1]
    while line[start_column] == ' ' and start_column < len(line):
        start_column += 1

    return ((start_line, start_column), (end_line, end_column - 1))

def _enrich_from_audits(report_: richreports.report, atok) -> richreports.report:
    for a in ast.walk(atok.tree):
        r = _audits(a, 'rules') 
        t = _audits(a, 'types')

        if isinstance(a, (ast.Assign, ast.AnnAssign)):
            target = a.targets[0] if hasattr(a, 'targets') else a.target
            (start, end) = _locations(report_, atok, target)
            if isinstance(r, SyntaxRestriction):
                _enrich_syntaxrestriction(report_, r, start, end)
            else:
                _enrich_from_type(report_, t, start, end)
                t_str = (
                    _type_to_str(t)
                    if not isinstance(t, TypeError) else
                    'TypeError: ' + str(t)
                )
                report_.enrich(
                    start, end,
                    '<span class="detail" data-detail="' + t_str + '">', '</span>',
                    True
                )

        elif isinstance(a, ast.Return):
            (start, end) = _locations(report_, atok, a)
            if isinstance(r, SyntaxRestriction):
                _enrich_syntaxrestriction(report_, r, start, end)
            else:
                _enrich_keyword(report_, start, 6)
                (start, end) = _locations(report_, atok, a.value)
                _enrich_from_type(report_, _audits(a.value, 'types'), start, end)

        elif isinstance(a, ast.For):
            (start, end) = _locations(report_, atok, a)
            _enrich_keyword(report_, start, 3)

        elif isinstance(a, ast.FunctionDef):
            (start, end) = _locations(report_, atok, a)
            _enrich_keyword(report_, start, 3)
            if isinstance(r, SyntaxRestriction):
                _enrich_syntaxrestriction(report_, r, start, end)

        elif isinstance(a, ast.Call):
            (start, end) = _locations(report_, atok, a.func)
            if isinstance(r, SyntaxRestriction):
                _enrich_syntaxrestriction(report_, r, start, end)
            else:
                _enrich_from_type(report_, t, start, end)
                if isinstance(t, TypeError):
                    t_str = 'TypeError: ' + str(t)
                    report_.enrich(
                        start, end,
                        '<span class="detail" data-detail="' + t_str + '">', '</span>',
                        True
                    )

        elif isinstance(a, ast.BinOp):
            t = TypeError('type cannot be determined') if t is None else t
            (_, (end_line, end_column)) = _locations(report_, atok, a.left)
            ((start_line, start_column), _) = _locations(report_, atok, a.right)
            (start, end) = (
                (end_line, end_column + 1),
                (start_line, start_column - 1)
            )
            if isinstance(r, SyntaxRestriction):
                _enrich_syntaxrestriction(report_, r, start, end)
            else:
                _enrich_from_type(report_, t, start, end)
                t_str = _type_to_str(t)
                report_.enrich(
                    start, end,
                    '<span class="detail" data-detail="' + t_str + '">', '</span>',
                    True
                )

        elif isinstance(a, ast.Name):
            (start, end) = _locations(report_, atok, a)
            if isinstance(r, SyntaxRestriction):
                _enrich_syntaxrestriction(report_, r, start, end)
            else:
                if t is not None:
                    _enrich_from_type(report_, t, start, end)
                    t_str = (
                        t.__name__
                        if not isinstance(t, TypeError) else
                        'TypeError: ' + str(t)
                    )
                    report_.enrich(
                        start, end,
                        '<span class="detail" data-detail="' + t_str + '">', '</span>',
                        True
                    )

        elif isinstance(a, ast.Constant):
            t = TypeError('type cannot be determined') if t is None else t
            (start, end) = _locations(report_, atok, a)
            if isinstance(r, SyntaxRestriction):
                _enrich_syntaxrestriction(report_, r, start, end)
            else:
                _enrich_from_type(report_, t, start, end)
                t_str = (
                    t.__name__
                    if not isinstance(t, TypeError) else
                    'TypeError: ' + str(t)
                )
                report_.enrich(
                    start, end,
                    '<span class="detail" data-detail="' + t_str + '">', '</span>',
                    True
                )

def audit(source: str) -> richreports.report:
    """
    Take a Python source file representing a Nada program, statically analyze
    it, and generate a report detailing the results.
    """
    (atok, skips) = _parse(source)
    root = atok.tree

    # Perform the static analyses.
    rules(root)
    types(root)

    # Perform the abstract execution.
    #root.body.append(ast.Expr(ast.Call(ast.Name('nada_main', ast.Load()), [], [])))
    #ast.fix_missing_locations(root)
    #exec(compile(root, path, 'exec'))

    # Add the results of the analyses to the report and ensure each line is
    # wrapped as an HTML element.
    report_ = richreports.report(source, line=1, column=0)
    _enrich_from_audits(report_, atok)
    for (i, line) in enumerate(report_.lines):
        if i in skips:
            report_.enrich(
                (i + 1, 0), (i + 1, len(line) - 1),
                '<span class="rules-SyntaxError">', '</span>',
                skip_whitespace=True
            )
            report_.enrich(
                (i + 1, 0), (i + 1, len(line) - 1),
                '<span class="detail" data-detail="SyntaxError">', '</span>',
                skip_whitespace=True
            )
        report_.enrich((i + 1, 0), (i + 1, len(line) - 1), '    <div>', '</div>')

    return report_

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs=1, help='Nada source file path')
    args = parser.parse_args()
    path = args.paths[0]

    with open(path, 'r', encoding='UTF-8') as file:
        source = file.read()
        report_ = audit(source)

    with open(path[:-2] + 'html', 'w') as file:
        head = ''
        file.write(
            '<html>' +
            head +
            '  <body>\n    <div id="detail"></div>\n' +
            report_.render() +
            '\n  </body>\n</html>\n'
        )

if __name__ == '__main__':
    _main()
